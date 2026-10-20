"""Scale-sharing shape sweep: output error vs bits/weight for (bc cols x br rows) tiles."""
import json, sys, time
import torch
from saq import colwise, gptq_bcq
from saq.model import WORK
torch.backends.cuda.matmul.allow_tf32 = False
SHAPES = [(64, 1), (256, 1), (8, 8), (8, 32), (16, 16), (32, 8), (8, 64), (16, 64), (64, 64), (4, 64), (1, 64), (8, 512)]
files = sys.argv[1].split(",")
qs = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3]
for f in files:
    dd = torch.load(WORK / "data/lab" / f); W, H = dd["W"].cuda().float(), dd["H"].cuda().float()
    R = W.shape[0]
    for q in qs:
        for bc, br in SHAPES:
            if R % br: continue
            t = time.time()
            if br == 1:
                r = gptq_bcq.quantize(W, H, q, G=bc, act_order=True)
                bpw = (q * 32 + q * (256 // bc) + 2) * 8 / 256
            else:
                r = colwise.fit_block(W, H, q, bc=bc, br=br)
                bpw = colwise.bpw_block(q, bc, br)
            print(json.dumps(dict(file=f[:-3], q=q, bc=bc, br=br, bpw=round(bpw, 4), err=r["err"], sec=round(time.time() - t))), flush=True)
