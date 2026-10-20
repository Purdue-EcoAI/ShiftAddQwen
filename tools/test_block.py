"""Block layout checks: fit_block -> pack -> byte dequant equality; BlockBCQLinear weight equality; refit descent."""
import torch
from saq import colwise
from saq.qlinear import BlockBCQLinear
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
R, C = 256, 1024
X = torch.randn(4096, C, device="cuda") * (torch.rand(C, device="cuda") + 0.1)
H = X.T @ X / 4096
W = torch.randn(R, C, device="cuda") * 0.05
for q, bc, br in [(3, 1, 64), (3, 8, 96), (3, 16, 48), (3, 64, 12), (3, 8, 512)]:
    if (q * (256 // bc)) % br:  # layout constraint: the table slice per row must be whole bytes
        print(f"q={q} {bc}x{br}: skipped (q*256/bc = {q * 256 // bc} not divisible by br)"); continue
    r0 = colwise.fit_block(W, H, q, bc=bc, br=br, refit_rounds=0)
    r = colwise.fit_block(W, H, q, bc=bc, br=br, refit_rounds=2)
    blk = colwise.pack_block(r["idx"].cpu(), r["T"].cpu(), r["d"].cpu(), q, bc, br)
    deq = colwise.dequant_block_bytes(blk, q, bc, br, C)
    m = BlockBCQLinear(r["idx"], r["T"].float(), r["d"], q, bc, br).cuda()
    w = m.weight()
    print(f"q={q} {bc}x{br}: err norefit {r0['err']:.5f} refit {r['err']:.5f} | pack max diff {(deq - r['Q'].cpu()).abs().max().item():.2e} | "
          f"module max diff {(w - r['Q']).abs().max().item():.2e} | bpw {blk.nbytes * 8 / (R * C):.3f} vs {colwise.bpw_block(q, bc, br):.3f}")
