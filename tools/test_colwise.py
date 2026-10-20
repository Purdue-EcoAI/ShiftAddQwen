"""Unit checks for saq.colwise: refit optimality vs brute force, packing round-trip, fit on a real matrix."""
import time

import numpy as np
import torch

from saq import colwise
from saq.model import WORK

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
R, C, q = 128, 512, 3
X = torch.randn(2048, C, device="cuda") * (torch.rand(C, device="cuda") + 0.1)
H = X.T @ X / 2048
W = torch.randn(R, C, device="cuda") * 0.05
r = colwise.fit(W, H, q, refit_rounds=0)
e0 = r["err"]
A, d = colwise.refit(W, H, r["idx"], r["A"], r["d"], q, rounds=1)
Q1 = colwise.dequant(r["idx"], A, d, q)
e1 = colwise.rel_err(W, Q1, H)
# brute-force check of step (1): table for group 0 with d fixed should be the LS optimum
S = colwise.signs(r["idx"], q)
g = 0
Sd = (S[:64] * r["d"][:64].repeat_interleave(256, 1)[:, :, None]).reshape(64, C * q)
Afull = torch.zeros(64, C, C * q, device="cuda")
col = (torch.arange(C, device="cuda") * q)[:, None] + torch.arange(q, device="cuda")[None]
Afull.scatter_(2, col[None].expand(64, -1, -1), Sd.view(64, C, q))
M = sum(Afull[i].T @ H @ Afull[i] for i in range(64))
rhs = sum(Afull[i].T @ H @ W[i] for i in range(64))
c = torch.linalg.solve(M + 1e-6 * M.diagonal().mean() * torch.eye(C * q, device="cuda"), rhs).view(C, q)
Aref = c.view(C // 256, 256, q).permute(0, 2, 1)
A1, _ = colwise.refit(W, H, r["idx"], r["A"], r["d"], q, rounds=1)
print("err before refit %.5f after %.5f" % (e0, e1))
print("table step matches brute force:", torch.allclose(A1[0].clamp_min(0), Aref.clamp_min(0), atol=1e-4, rtol=1e-3))
r2 = colwise.fit(W, H, q)
blk = colwise.pack_blocks(r2["idx"].cpu(), r2["A"].cpu(), r2["d"].cpu(), q)
deq = colwise.dequant_blocks(blk, q, C)
print("pack round-trip max diff:", (deq - r2["Q"].cpu()).abs().max().item(), "bytes/weight", blk.nbytes / (R * C))
f = WORK / "data/lab/L15.self_attn.q_proj.pt"
if f.exists():
    dd = torch.load(f)
    Wr, Hr = dd["W"].cuda().float(), dd["H"].cuda().float()
    from saq import gptq_bcq
    for qq in (2, 3):
        t = time.time(); row = gptq_bcq.quantize(Wr, Hr, qq, G=64, act_order=True)["err"]; t1 = time.time() - t
        t = time.time(); col0 = colwise.fit(Wr, Hr, qq, refit_rounds=0)["err"]; t2 = time.time() - t
        t = time.time(); col2 = colwise.fit(Wr, Hr, qq, refit_rounds=2)["err"]; t3 = time.time() - t
        print(f"q_proj q={qq}: row+refit {row:.5f} ({t1:.0f}s)  col {col0:.5f} ({t2:.0f}s)  col+refit {col2:.5f} ({t3:.0f}s)")
