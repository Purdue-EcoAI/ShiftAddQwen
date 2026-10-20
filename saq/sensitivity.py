"""Fisher-weighted per-linear sensitivity for mixed-bit allocation, plus the knapsack planner.

  python -m saq.sensitivity measure        # -> work/results/sensitivity.json
  python -m saq.sensitivity plan 2.75      # -> work/plans/plan_2.75.json (avg bpw over decoder linears)

Loss change from quantizing linear l (y = W x), second-order with independent rows:
    dL ~ sum_r E[g_r^2] * (dW_r H dW_r^T),   g = dL/dy on calibration text (CE loss)
E[g_r^2] comes from backward hooks on the bf16 model; dW_r H dW_r^T from quantizing a random
row subset of every linear at 2/3/4 bits with the same BCQ+GPTQ quantizer used for real.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from . import bcq, data, gptq_bcq
from .model import MODEL, WORK, gguf_col_perm, load, quant_linears, text_layers

torch.backends.cuda.matmul.allow_tf32 = False
OUT = WORK / "results" / "sensitivity.json"
BPW = {2: bcq.bits_per_weight(2, 64, "u8", False), 3: bcq.bits_per_weight(3, 64, "u8", False),
       4: bcq.bits_per_weight(4, 64, "u8", False)}


def fisher(model, seqs):
    layers = text_layers(model)
    g2, hooks = {}, []
    for li, layer in enumerate(layers):
        for n, m in quant_linears(layer).items():
            key = f"model.language_model.layers.{li}.{n}"

            def fwd(mod, inp, out, key=key):
                if not out.requires_grad:  # first (no-grad) pass of a checkpointed layer
                    return

                def bwd(g):
                    gg = g.detach().float().reshape(-1, g.shape[-1])
                    g2[key] = g2.get(key, 0) + (gg ** 2).sum(0)
                out.register_hook(bwd)
            hooks.append(m.register_forward_hook(fwd))
    lm = model.model.language_model
    ntok = 0
    for s in seqs:
        ids = s[None].cuda()
        emb = lm.embed_tokens(ids).detach().requires_grad_(True)
        h = lm(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
        loss = 0.0
        for c0 in range(0, h.shape[0] - 1, 512):
            c1 = min(c0 + 512, h.shape[0] - 1)
            logits = model.lm_head(h[c0:c1]).float()
            loss = loss + F.cross_entropy(logits, ids[0, c0 + 1:c1 + 1], reduction="sum")
        loss.backward()
        ntok += ids.shape[1] - 1
    for h in hooks:
        h.remove()
    return {k: (v / ntok).cpu() for k, v in g2.items()}


def hessians(model, batches, li_set):
    layers = text_layers(model)
    Hs, cnt, hooks = {}, {}, []
    for li in li_set:
        for n, m in quant_linears(layers[li]).items():
            key = f"model.language_model.layers.{li}.{n}"

            def hook(mod, inp, key=key):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                if key not in Hs:
                    Hs[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                    cnt[key] = 0
                Hs[key].addmm_(x.T, x)
                cnt[key] += x.shape[0]
            hooks.append(m.register_forward_pre_hook(hook))
    with torch.no_grad():
        for b in batches:
            model.model.language_model(input_ids=b.cuda(), use_cache=False)
    for h in hooks:
        h.remove()
    return {k: Hs[k] / cnt[k] for k in Hs}


def row_err(W, H, Q):
    D = W - Q
    return torch.einsum("rc,cd,rd->r", D, H, D)


def measure(sub_rows=768, nfisher=24, nhess=32):
    tok, model = load(MODEL)
    calib = data.calibration(tok, 128, 2048)
    model.model.visual = None
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.train()
    f = fisher(model, calib[:nfisher])
    model.eval()
    torch.cuda.empty_cache()
    res = {}
    layers = text_layers(model)
    batches = list(calib[:nhess].split(4))
    g = torch.Generator().manual_seed(0)
    for l0 in range(0, len(layers), 4):
        Hs = hessians(model, batches, range(l0, min(l0 + 4, len(layers))))
        for key, H in Hs.items():
            li = int(key.split(".")[3])
            n = ".".join(key.split(".")[4:])
            W = quant_linears(layers[li])[n].weight.data.float()
            cp = gguf_col_perm(n, W.shape[1])
            if cp is not None:
                cp = cp.cuda()
                W, H = W[:, cp], H[cp][:, cp]
            R = W.shape[0]
            rows = torch.randperm(R, generator=g)[:min(sub_rows, R)].cuda()
            Ws, fr = W[rows], f[key].cuda()[rows]
            ent = {"params": W.numel(), "shape": list(W.shape)}
            Hd = H.clone()
            Hd.diagonal().add_(0.01 * H.diagonal().mean())
            for q in (2, 3, 4):
                Q = gptq_bcq.quantize(Ws, H, q, G=64, act_order=True)["Q"]
                e = row_err(Ws, Hd, Q)
                ent[f"S{q}"] = (fr * e).sum().item() * R / len(rows)
                ent[f"rel{q}"] = (e.sum() / torch.einsum("rc,cd,rd->", Ws, Hd, Ws)).item()
            res[key] = ent
            print(key, json.dumps({k: v for k, v in ent.items() if k != "shape"}), flush=True)
        del Hs
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(res, indent=1))


# relative error growth of larger alpha groups vs G=64 (layer_lab, mean over 5 matrices)
G_RATIO = {(2, 128): 1.085, (2, 256): 1.14, (3, 128): 1.10, (3, 256): 1.17, (4, 128): 1.12, (4, 256): 1.21,
           (5, 128): 1.13, (5, 256): 1.23, (6, 128): 1.14, (6, 256): 1.25}  # 5/6: extrapolated trend


def plan_qg(avg_bpw, sens, groups=(64, 128, 256), qs=(2, 3, 4, 5, 6)):
    """Knapsack over (bits, group size). Returns {name: [q, G]} and achieved avg bpw."""
    opts = [(q, G) for q in qs for G in groups]
    cost = lambda k, q, G: sens[k][f"S{q}"] * G_RATIO.get((q, G), 1.0)
    bpw = lambda q, G: bcq.bits_per_weight(q, G, "u8", False)
    keys = list(sens)
    P = sum(sens[k]["params"] for k in keys)

    def choose(lam):
        out, bits = {}, 0.0
        for k in keys:
            q, G = min(opts, key=lambda o: cost(k, *o) + lam * sens[k]["params"] * bpw(*o))
            out[k] = [q, G]
            bits += sens[k]["params"] * bpw(q, G)
        return out, bits

    lo, hi = 0.0, 1.0
    while choose(hi)[1] > avg_bpw * P:
        hi *= 10
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if choose(mid)[1] > avg_bpw * P else (lo, mid)
    out, bits = choose(hi)
    return out, bits / P


def plan(avg_bpw, sens=None, qset=(2, 3, 4), min_bits=None):
    """Lagrangian knapsack: minimise sum S_l(q_l) s.t. sum params_l * bpw(q_l) <= avg_bpw * sum params_l."""
    sens = sens or json.loads(OUT.read_text())
    keys = list(sens)
    P = sum(sens[k]["params"] for k in keys)
    budget = avg_bpw * P

    def choose(lam):
        out, bits = {}, 0.0
        for k in keys:
            cands = [q for q in qset if not (min_bits and q < min_bits.get(k, 0))]
            q = min(cands, key=lambda q: sens[k][f"S{q}"] + lam * sens[k]["params"] * BPW[q])
            out[k] = q
            bits += sens[k]["params"] * BPW[q]
        return out, bits

    lo, hi = 0.0, 1.0
    while choose(hi)[1] > budget:
        hi *= 10
    for _ in range(100):
        mid = (lo + hi) / 2
        if choose(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    out, bits = choose(hi)
    return out, bits / P


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure()
    else:
        avg = float(sys.argv[2])
        src = sys.argv[3] if len(sys.argv) > 3 else "fisher"
        sens = None
        if src.startswith("kld"):  # merge direct end-to-end shards (saq.sens_kld)
            sens = {}
            for f in sorted((WORK / "results" / "sens_kld").glob("L*.json")):
                sens.update(json.loads(f.read_text()))
            FLOOR = 3.0e-4  # bf16 forward noise floor: any perturbation of one linear costs ~3e-4 KLD
            for k, v in sens.items():  # remove floor, enforce monotone positive costs
                for q in (2, 3, 4):
                    v[f"S{q}"] = v[f"S{q}"] - FLOOR
                v["S4"] = max(v["S4"], 1e-6)
                v["S3"] = max(v["S3"], v["S4"])
                v["S2"] = max(v["S2"], v["S3"])
                # 5/6 planes cost less than the bf16 floor to measure directly: extrapolate with this
                # linear's own per-plane decay, taken from the cleanly measured 2->3 bit step
                r = min(max(v["S3"] / v["S2"], 0.25), 0.5)
                v["S5"] = v["S4"] * r
                v["S6"] = v["S5"] * r
            assert len(sens) == 200, len(sens)
        p, real = plan_qg(avg, sens) if src == "kldG" else plan(avg, sens)
        d = WORK / "plans"
        d.mkdir(exist_ok=True)
        (d / f"plan_{src}_{avg}.json").write_text(json.dumps(p, indent=1))
        from collections import Counter
        print(f"avg bpw {real:.4f}", Counter(str(v) for v in p.values()))
