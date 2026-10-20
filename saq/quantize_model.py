"""Quantize Qwen3.5-9B decoder linears with output-aware BCQ (ShiftAddLLM style), layer by layer.

python -m saq.quantize_model --out work/runs/sa3_g64 --bits 3 --group 64 [--plan plan.json]

Every decoder layer is calibrated on inputs produced by the already-quantized earlier layers.
Saves per-linear codes/alphas (for GGUF export) + a json log, then reports PyTorch KLD vs bf16.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import colwise, data, gptq_bcq
from .model import (MODEL, WORK, eval_windows, gguf_col_perm, reference_cache, SHARED_INPUT, STAGES, kld_eval, load,
                    quant_linears, reference_logprobs,
                    run_until, text_layers)

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--plan", help="json {full_linear_name: bits} overriding --bits")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--fmt", default="u8", choices=["u8", "pot", "fp16"])
    ap.add_argument("--mu", action="store_true", help="per-group offset")
    ap.add_argument("--nseq", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--no-hweight", action="store_true")
    ap.add_argument("--act-order", action="store_true")
    ap.add_argument("--true-seq", action="store_true")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=None, help="only first N layers (debug)")
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--calib", type=int, default=1, help="calibration mix version (2 adds multilingual + math)")
    ap.add_argument("--method", default="bcq", choices=["bcq", "uniform", "rtn"],
                    help="bcq: ShiftAdd binary planes; uniform: GPTQ with asymmetric uniform groups (baseline); "
                         "rtn: uniform round-to-nearest (baseline). Baselines are fake-quant only (no export).")
    ap.add_argument("--eval-texts", default="wiki", help="comma list of eval texts for the PyTorch KLD: wiki,mix")
    ap.add_argument("--layout", default="row", choices=["row", "block"],
                    help="row: scales per (row, G input cols) [sa* types]; block: per (bc cols x br rows) tile [sab* types]")
    ap.add_argument("--tile", default="8x32", help="block layout default tile bcxbr (plan entries [q, bc, br] override)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    plan = json.loads(Path(a.plan).read_text()) if a.plan else {}

    tok, model = load(a.model)
    texts = a.eval_texts.split(",")
    evals = {t: (eval_windows(tok, t, 512, a.chunks), None) for t in texts}
    evals = {t: (w, reference_logprobs(model, w, reference_cache(t, a.chunks))) for t, (w, _) in evals.items()}
    calib = data.calibration(tok, a.nseq, a.seqlen, version=a.calib)
    batches = list(calib.split(a.bs))
    layers = text_layers(model)
    nl = a.layers or len(layers)
    log = []
    t0 = time.time()
    for li in range(nl):
        lins = quant_linears(layers[li])
        stages = [tuple(n for n in s if n in lins) for s in STAGES] if a.true_seq else [tuple(lins)]
        for stage in stages:
            stage = [n for n in stage if n]
            if not stage:
                continue
            # one Hessian per distinct input
            rep = {}
            for n in stage:
                rep[n] = n
                for grp in SHARED_INPUT:
                    if n in grp:
                        rep[n] = next(g for g in grp if g in lins)
            Hs, counts = {}, {}

            def mk(name):
                def hook(mod, inp):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    if name not in Hs:
                        Hs[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                        counts[name] = 0
                    Hs[name].addmm_(x.T, x)
                    counts[name] += x.shape[0]
                return hook

            reps = sorted(set(rep.values()))
            run_until(model, li, batches, [(lins[r], mk(r)) for r in reps])
            for n in stage:
                lin = lins[n]
                H = Hs[rep[n]] / counts[rep[n]]
                full = f"model.language_model.layers.{li}.{n}"
                q, G = plan.get(full, a.bits), a.group
                bc, br = (int(x) for x in a.tile.split("x"))
                if isinstance(q, list):  # [bits, group] (row layout) or [bits, bc, br] (block layout)
                    if len(q) == 3:
                        q, bc, br = q
                    else:
                        q, G = q
                ts = time.time()
                W = lin.weight.data
                cp = gguf_col_perm(n, W.shape[1])
                if cp is not None:  # quantize in llama.cpp's column order so groups match the GGUF layout
                    cp = cp.to(W.device)
                    W, H = W[:, cp], H[cp][:, cp]
                if a.method == "uniform":
                    res = gptq_bcq.quantize_uniform(W, H, q, G=G, percdamp=a.percdamp, act_order=a.act_order)
                elif a.method == "rtn":
                    res = gptq_bcq.rtn_uniform(W, H, q, G=G)
                elif a.layout == "block":
                    res = colwise.fit_block(W, H, q, bc=bc, br=br, percdamp=a.percdamp, act_order=a.act_order)
                else:
                    res = gptq_bcq.quantize(W, H, q=q, G=G, fmt=a.fmt, mu_on=a.mu,
                                            percdamp=a.percdamp, hweight=not a.no_hweight, rounds=a.rounds,
                                            refit=not a.no_refit, act_order=a.act_order)
                Qw = res["Q"]
                if cp is not None:
                    Qw = Qw[:, cp.argsort()]
                lin.weight.data.copy_(Qw.to(lin.weight.dtype))
                rec = {"name": full, "bits": q, "shape": list(lin.weight.shape), "rel_err": res["err"],
                       "sec": round(time.time() - ts, 2), "layout": a.layout, "method": a.method,
                       "tile": f"{bc}x{br}" if a.layout == "block" else f"G{G}"}
                log.append(rec)
                print(json.dumps(rec), flush=True)
                if a.method != "bcq":
                    pass  # baselines are evaluated in PyTorch only
                elif not a.no_save and a.layout == "block":
                    torch.save({"layout": "block", "idx": res["idx"].cpu(), "q": q, "bc": bc, "br": br,
                                "T": res["T"].to(torch.uint8).cpu(), "d": res["d"].cpu(),
                                "col_perm_applied": cp is not None}, out / f"{full}.pt")
                elif not a.no_save:
                    st = res["store"]
                    torch.save({"idx": res["idx"].to(torch.uint8).cpu(), "q": q, "G": G, "fmt": a.fmt,
                                "alpha": st["alpha"].cpu(), "col_perm_applied": cp is not None, "mu": None if st["mu"] is None else st["mu"].cpu(),
                                "alpha_store": {k: v.cpu() for k, v in st["alpha_store"].items()},
                                "mu_store": {k: v.cpu() for k, v in st.get("mu_store", {}).items()}},
                               out / f"{full}.pt")
            del Hs
            torch.cuda.empty_cache()
        print(f"layer {li} done, {time.time() - t0:.0f}s", flush=True)
    res = {}
    for t, (w, ref) in evals.items():
        r = kld_eval(model, w, ref)
        res[t] = r
        print("EVAL", t, json.dumps(r), flush=True)
    res["quant_sec"] = time.time() - t0
    (out / "log.json").write_text(json.dumps({"layers": log, "eval": res}, indent=1))


if __name__ == "__main__":
    main()
