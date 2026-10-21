"""End-to-end distillation of the BCQ scales (codes frozen) — the "E2E-QP" idea applied to ShiftAdd.

python -m saq.finetune --run work/runs/X --out work/runs/X_ft [--steps N --lr 2e-5]

Teacher = bf16 Qwen3.5-9B, targets = its top-k log-probs on held-out calibration-style text.
Student = same model with every quantized linear replaced by BCQLinear (trainable alpha only,
snapped to the u8 storage grid via STE). Saves updated per-linear .pt files compatible with
export_gguf.py, then reports PyTorch KLD on wikitext-2 test.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import data
from .model import MODEL, WORK, gguf_col_perm, kld_eval, load, reference_logprobs, text_layers
from .qlinear import BCQLinear, BlockBCQLinear, UniformLinear


@torch.no_grad()
def teacher_topk(model, seqs, k, cache):
    if Path(cache).exists():
        return torch.load(cache)
    vals, inds = [], []
    for s in seqs:
        out = model.model.language_model(input_ids=s[None].cuda(), use_cache=False).last_hidden_state
        for c in out[0].split(512):
            lp = F.log_softmax(model.lm_head(c).float(), -1)
            v, i = lp.topk(k, -1)
            vals.append(v.half().cpu())
            inds.append(i.int().cpu())
    res = {"v": torch.cat(vals), "i": torch.cat(inds)}
    torch.save(res, cache)
    return res


def swap_in(model, run, gran="group"):
    layers = text_layers(model)
    qmods = []
    for f in sorted(Path(run).glob("model.language_model.layers.*.pt")):
        st = torch.load(f)
        parts = f.stem.split(".")
        li, sub = int(parts[3]), ".".join(parts[4:6])
        parent = layers[li].get_submodule(parts[4])
        old = getattr(parent, parts[5])
        cp = gguf_col_perm(sub, old.in_features) if st.get("col_perm_applied") else None
        if st.get("layout") == "uniform":
            m = UniformLinear(st["idx"].cuda(), st["lo"].cuda(), st["scale"].cuda(), st["q"], st["G"], cp)
        elif st.get("layout") == "block":
            m = BlockBCQLinear(st["idx"].cuda(), st["T"].cuda(), st["d"].cuda(), st["q"], st["bc"], st["br"], cp)
        else:
            m = BCQLinear(st["idx"].cuda(), st["alpha"].cuda(), st["q"], st["G"],
                          None if st["mu"] is None else st["mu"].cuda(), cp, st["fmt"], gran=gran)
        setattr(parent, parts[5], m)
        del old
        qmods.append((f.stem, m, st))
    torch.cuda.empty_cache()
    return qmods


def student_loss(model, ids, tv, ti, chunk=512):
    """KL(teacher || student) per token over the teacher's top-k tokens plus one 'rest of vocab' bucket
    (no renormalisation, so the student is not pushed to be over-confident)."""
    lm = model.model.language_model
    h = lm.embed_tokens(ids)
    # replicate Qwen3_5TextModel.forward with per-layer checkpointing
    pos = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(4, 1, -1)
    text_pos, mpos = pos[0], pos[1:]
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import create_recurrent_attention_mask
    mk = dict(config=lm.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=text_pos)
    masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
    pe = lm.rotary_emb(h, mpos)
    for i, layer in enumerate(lm.layers):
        h = checkpoint(layer, h, pe, masks[lm.config.layer_types[i]], text_pos, use_reentrant=False)
    h = lm.norm(h)[0]
    tlp = tv.float()  # teacher log-probs of its top-k (true, unnormalised over k)
    total = 0.0
    n = h.shape[0]

    def chunk_loss(hc, tlpc, tic):
        lp = F.log_softmax(model.lm_head(hc).float(), -1)
        slp = lp.gather(-1, tic)
        tp = tlpc.exp()
        t_rest = (1 - tp.sum(-1)).clamp_min(1e-8)
        s_rest = (1 - slp.exp().sum(-1)).clamp_min(1e-8)
        kl = (tp * (tlpc - slp)).sum(-1) + t_rest * (t_rest.log() - s_rest.log())
        return kl.sum()

    for s in range(0, n, chunk):
        total = total + checkpoint(chunk_loss, h[s:s + chunk], tlp[s:s + chunk], ti[s:s + chunk].long(),
                                   use_reentrant=False)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nseq", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--train-norms", action="store_true")
    ap.add_argument("--gran", default="group", choices=["group", "row"])
    ap.add_argument("--calib-seed", type=int, default=1)
    ap.add_argument("--calib", type=int, default=1, help="data mix version (2 adds multilingual + math)")
    ap.add_argument("--chunks", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ft_args.json").write_text(json.dumps(vars(a), indent=1))

    tok, model = load(a.model)
    windows = data.wiki_test_windows(tok, 512, a.chunks)
    ref = reference_logprobs(model, windows, WORK / f"data/ref_lp_{a.chunks}.pt")
    seqs = data.calibration(tok, a.nseq, a.seqlen, seed=a.calib_seed, version=a.calib)  # disjoint seed
    v = "" if a.calib == 1 else f"v{a.calib}_"
    tgt = teacher_topk(model, seqs, a.topk,
                       WORK / f"data/teacher_{v}top{a.topk}_{a.nseq}x{a.seqlen}_s{a.calib_seed}.pt")

    model.model.visual = None
    torch.cuda.empty_cache()
    qmods = swap_in(model, a.run, a.gran)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for _, m, _ in qmods:
        m.theta.requires_grad_(True)
        params.append(m.theta)
        if m.mu is not None:
            m.mu.requires_grad_(True)
            params.append(m.mu)
    norm_params = []
    if a.train_norms:
        for n, p in model.model.language_model.named_parameters():
            if n.endswith("layernorm.weight") or n == "norm.weight":
                p.requires_grad_(True)
                norm_params.append(p)
    print("trainable alpha params:", sum(p.numel() for p in params) / 1e6, "M", flush=True)
    before = kld_eval(model, windows, ref)
    print("BEFORE", json.dumps(before), flush=True)

    groups = [{"params": params, "lr": a.lr}]
    if norm_params:
        groups.append({"params": norm_params, "lr": a.lr * 0.1})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    steps = int(a.epochs * a.nseq)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    for _, m, _ in qmods:
        m.snap = False  # train continuous scales; snapped to the storage grid at the end
    model.train()
    t0 = time.time()
    L = a.seqlen
    ema = None
    for step in range(steps):
        si = step % a.nseq
        ids = seqs[si][None].cuda()
        tv = tgt["v"][si * L:(si + 1) * L].cuda()
        ti = tgt["i"][si * L:(si + 1) * L].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = student_loss(model, ids, tv, ti)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + norm_params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step}/{steps} loss {loss.item():.4f} ema {ema:.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    for _, m, _ in qmods:
        m.snap = True
    after = kld_eval(model, windows, ref)
    print("AFTER", json.dumps(after), flush=True)

    # save updated per-linear files (export-compatible)
    for name, m, st in qmods:
        st = dict(st)
        if isinstance(m, BlockBCQLinear):
            st["T"], st["d"] = (x.cpu() for x in m.export())
            torch.save(st, out / f"{name}.pt")
            continue
        if isinstance(m, UniformLinear):
            st["lo"], st["scale"] = (x.cpu() for x in m.export())
            torch.save(st, out / f"{name}.pt")
            continue
        aq, store = m.export_alpha()
        st["alpha"] = aq.cpu()
        st["alpha_store"] = {k: v.cpu() for k, v in store.items()}
        if m.mu is not None:
            st["mu"] = m.mu.detach().cpu()
        torch.save(st, out / f"{name}.pt")
    if norm_params:
        torch.save({n: p.detach().cpu() for n, p in model.model.language_model.named_parameters()
                    if n.endswith("layernorm.weight") or n == "norm.weight"}, out / "norms.pt")
    for f in ("args.json", "plan.json"):
        if (Path(a.run) / f).exists():
            shutil.copy(Path(a.run) / f, out / f)
    (out / "ft_log.json").write_text(json.dumps({"before": before, "after": after, "sec": time.time() - t0}))


if __name__ == "__main__":
    main()
