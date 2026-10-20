import torch, time
from saq import gptq_bcq, bcq
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
R, C, G, q = 64, 1024, 64, 3
X = torch.randn(4096, C, device="cuda") * torch.rand(C, device="cuda")
H = X.T @ X / 4096
W = torch.randn(R, C, device="cuda")
idx = torch.randint(0, 2 ** q, (R, C), device="cuda")
def old(W0, H, idx, q, G, mu_on):
    R, C = W0.shape; ng = C // G; k = q + mu_on; n = ng * k
    S = bcq.codes_to_signs(idx, q)
    if mu_on: S = torch.cat([S, torch.ones_like(S[..., :1])], -1)
    col = (torch.arange(C, device=W0.device) // G * k)[:, None] + torch.arange(k, device=W0.device)[None, :]
    A = torch.zeros(R, C, n, device=W0.device); A.scatter_(2, col[None].expand(R, -1, -1), S)
    M = A.transpose(1, 2) @ torch.matmul(H, A); rhs = (A * (W0 @ H)[:, :, None]).sum(1)
    M = M + (1e-5 * M.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] + 1e-10) * torch.eye(n, device=W0.device)
    return torch.linalg.solve(M, rhs[..., None])[..., 0].view(R, ng, k)
for mu in (False, True):
    a_new, m_new = gptq_bcq.refit_alpha(W, H, idx, q, G, mu)
    x = old(W, H, idx, q, G, mu)
    print(mu, (a_new - x[..., :q]).abs().max().item(), x.abs().max().item())
d = torch.load("work/data/lab/L14.mlp.down_proj.pt"); W, H = d["W"].cuda().float(), d["H"].cuda().float()
t = time.time(); r = gptq_bcq.quantize(W, H, 3, G=64); print("down q3 err", r["err"], "sec", time.time() - t)
