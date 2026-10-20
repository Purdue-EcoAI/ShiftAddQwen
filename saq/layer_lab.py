"""Quick per-matrix ablations on real Qwen3.5-9B weights + calibration Hessians.

  python -m saq.layer_lab collect     # dump W and H for a few representative layers
  python -m saq.layer_lab run         # grid of quantizer variants -> relative output error table
"""
import itertools
import json
import sys
import time

import torch

from . import bcq, data, gptq_bcq
from .model import MODEL, WORK, SHARED_INPUT, load, quant_linears, run_until, text_layers

torch.backends.cuda.matmul.allow_tf32 = False
LAB = WORK / "data" / "lab"
LAYERS = [0, 3, 14, 15, 30, 31]  # GDN + full-attention layers, early / middle / late


def collect():
    tok, model = load(MODEL)
    calib = data.calibration(tok, 128, 2048)
    batches = list(calib[:64].split(8))
    LAB.mkdir(parents=True, exist_ok=True)
    layers = text_layers(model)
    for li in LAYERS:
        lins = quant_linears(layers[li])
        Hs, cnt = {}, {}

        def mk(name):
            def hook(mod, inp):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                Hs.setdefault(name, torch.zeros(x.shape[1], x.shape[1], device=x.device)).addmm_(x.T, x)
                cnt[name] = cnt.get(name, 0) + x.shape[0]
            return hook

        run_until(model, li, batches, [(m, mk(n)) for n, m in lins.items()])
        for n, m in lins.items():
            torch.save({"W": m.weight.data.cpu(), "H": (Hs[n] / cnt[n]).cpu()}, LAB / f"L{li}.{n}.pt")
            print("saved", li, n, tuple(m.weight.shape), flush=True)


VARIANTS = {
    # name: (fn, kwargs)  -- bits/weight computed per variant
    "rtn_uniform_g64":  ("uniform_rtn", dict(G=64)),
    "gptq_uniform_g64": ("uniform", dict(G=64)),
    "gptq_uniform_g32": ("uniform", dict(G=32)),
    "bcq_plain_g64":    ("bcq", dict(G=64, refit=False, hweight=False)),
    "bcq_hw_g64":       ("bcq", dict(G=64, refit=False)),
    "bcq_hw_refit_g64": ("bcq", dict(G=64)),
    "bcq_refit_r2_g64": ("bcq", dict(G=64, rounds=2)),
    "bcq_refit_g32":    ("bcq", dict(G=32)),
    "bcq_refit_g128":   ("bcq", dict(G=128)),
    "bcq_actord_g64":   ("bcq", dict(G=64, act_order=True)),
    "bcq_actord_g128":  ("bcq", dict(G=128, act_order=True)),
    "bcq_actord_g256":  ("bcq", dict(G=256, act_order=True)),
    "bcq_refit_mu_g64": ("bcq", dict(G=64, mu_on=True)),
    "bcq_refit_fp16a":  ("bcq", dict(G=64, fmt="fp16")),
    "bcq_refit_pot":    ("bcq", dict(G=64, fmt="pot")),
    "bcq_refit_actord": ("bcq", dict(G=64, act_order=True)),
}


def bpw(kind, q, kw):
    G = kw.get("G", 64)
    if kind.startswith("uniform"):
        return q + 32 / G
    return bcq.bits_per_weight(q, G, kw.get("fmt", "u8"), kw.get("mu_on", False))


def run(names=None, qs=(2, 3, 4), files=None):
    files = files or sorted(LAB.glob("*.pt"))
    rows = []
    for f in files:
        d = torch.load(f)
        W, H = d["W"].cuda().float(), d["H"].cuda().float()
        for q in qs:
            for name, (kind, kw) in VARIANTS.items():
                if names and name not in names:
                    continue
                t = time.time()
                if kind == "bcq":
                    e = gptq_bcq.quantize(W, H, q, **kw)["err"]
                elif kind == "uniform":
                    e = gptq_bcq.quantize_uniform(W, H, q, **kw)["err"]
                else:
                    Lq = []
                    G = kw["G"]
                    Wg = W.view(-1, G)
                    Lv = gptq_bcq._uniform_levels(Wg, q)
                    k = (Wg[..., None] - Lv[:, None, :]).abs().argmin(-1)
                    Q = torch.gather(Lv, 1, k).view_as(W)
                    Hd = H.clone()
                    Hd.diagonal().add_(0.01 * H.diagonal().mean())
                    e = gptq_bcq.rel_err(W, Q, Hd)
                r = dict(file=f.stem, q=q, variant=name, bpw=round(bpw(kind, q, kw), 4), rel_err=e,
                         sec=round(time.time() - t, 1))
                rows.append(r)
                print(json.dumps(r), flush=True)
    (WORK / "results").mkdir(exist_ok=True)
    out = WORK / "results" / "layer_lab.jsonl"
    with open(out, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect()
    else:
        names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else None
        qs = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (2, 3, 4)
        files = [LAB / f for f in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
        run(names, qs, files)
