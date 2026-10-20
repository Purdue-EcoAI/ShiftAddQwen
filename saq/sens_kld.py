"""Direct end-to-end sensitivity: KLD increase when ONE linear is quantized (BCQ+GPTQ) at 2/3/4 bits.

  python -m saq.sens_kld --layers 0-7      # shard -> work/results/sens_kld/L0-7.json
  python -m saq.sensitivity plan 3.0 --sens kld

Held-out text (calibration-style mix, seed 2) in 512-token windows, KLD vs the bf16 model over the
second half of each window. Additivity across linears is assumed by the knapsack planner.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data, gptq_bcq
from .model import MODEL, WORK, gguf_col_perm, load, quant_linears, text_layers, text_logits
from .sensitivity import hessians

torch.backends.cuda.matmul.allow_tf32 = False


@torch.no_grad()
def kld_windows(model, win, ref):
    ctx = win.shape[1]
    tot = 0.0
    for w, r in zip(win, ref):
        lp = F.log_softmax(text_logits(model, w[None].cuda())[0, ctx // 2 - 1:-1].float(), -1)
        r = r.cuda().float()
        tot += (r.exp() * (r - lp)).sum(-1).mean().item()
    return tot / len(win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--nwin", type=int, default=24)
    a = ap.parse_args()
    l0, l1 = map(int, a.layers.split("-"))
    out = WORK / "results" / "sens_kld"
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(MODEL)
    model.model.visual = None
    held = data.calibration(tok, 64, 2048, seed=2)
    win = held[:, :1024].reshape(-1, 512)[: a.nwin]
    ref = []
    with torch.no_grad():
        for w in win:
            ref.append(F.log_softmax(text_logits(model, w[None].cuda())[0, 255:-1].float(), -1).half().cpu())
    base = kld_windows(model, win, ref)  # ~0 (numerical)
    calib = data.calibration(tok, 128, 2048)
    batches = list(calib[:32].split(4))
    layers = text_layers(model)
    res = {}
    for c0 in range(l0, l1 + 1, 4):
        Hs = hessians(model, batches, range(c0, min(c0 + 4, l1 + 1)))
        for key, H in Hs.items():
            li = int(key.split(".")[3])
            n = ".".join(key.split(".")[4:])
            lin = quant_linears(layers[li])[n]
            W0 = lin.weight.data.clone()
            W, Hq = W0.float(), H
            cp = gguf_col_perm(n, W.shape[1])
            if cp is not None:
                cp = cp.cuda()
                W, Hq = W[:, cp], H[cp][:, cp]
            ent = {"params": W.numel()}
            for q in (2, 3, 4):
                Q = gptq_bcq.quantize(W, Hq, q, G=64, act_order=True)["Q"]
                if cp is not None:
                    Q = Q[:, cp.argsort()]
                lin.weight.data.copy_(Q.to(W0.dtype))
                ent[f"S{q}"] = kld_windows(model, win, ref) - base
            lin.weight.data.copy_(W0)
            res[key] = ent
            print(key, json.dumps(ent), flush=True)
            (out / f"L{l0}-{l1}.json").write_text(json.dumps(res, indent=1))
        del Hs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
