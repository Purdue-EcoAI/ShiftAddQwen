"""Output-error-aware BCQ (ShiftAddLLM's multi-objective idea) for one linear layer.

Pipeline for W [R, C] with input Hessian H = E[x x^T] [C, C]:

1. Column-wise GPTQ/OPTQ sweep: at the start of every input group of G columns, fit the
   per-row BCQ scales (alpha, optional offset mu) on the *error-compensated* weights,
   quantize alpha to its storage format, then assign every column to its nearest BCQ level
   and push the rounding error onto the not-yet-quantized columns via the inverse Hessian.
2. Alpha refit: with the binary codes fixed, the layer output is linear in alpha, so solve
   min_alpha  (w0 - A alpha)^T H (w0 - A alpha)   exactly, per output row.
3. Optionally alternate (GPTQ with frozen alphas -> refit) for a few rounds.

Returns codes, alphas, offsets and the dequantized weight.
"""
import math

import torch

from . import bcq


def _group_alpha_store(alpha, d, fmt):
    """Quantize alpha [R, q] using a fixed per-row super-block scale d [R]."""
    if fmt == "fp16":
        return alpha.half().float()
    if fmt == "u8":
        return torch.round(alpha / d[:, None]).clamp(0, 255) * d[:, None]
    if fmt == "pot":
        e = torch.round(-torch.log2((alpha / d[:, None]).clamp_min(2 ** -15))).clamp(0, 15)
        return d[:, None] * torch.exp2(-e)
    raise ValueError(fmt)


def _store_all(alpha, mu, fmt, mu_on, SB, G):
    """Quantize all alphas [R, ng, q] / mu [R, ng] with per-super-block scales."""
    R, ng, q = alpha.shape
    gps = SB // G
    a = alpha.view(R, ng // gps, gps, q)
    aq, sa = bcq.quant_alpha(a, fmt)
    out = {"alpha": aq.view(R, ng, q), "alpha_store": sa}
    if mu_on:
        m = mu.view(R, ng // gps, gps)
        mq, sm = bcq.quant_mu(m)
        out["mu"] = mq.view(R, ng)
        out["mu_store"] = sm
    else:
        out["mu"] = None
    return out


def robust_hinv(H, damp):
    """Upper Cholesky factor of H^-1; retries with growing damping if H is numerically indefinite."""
    assert torch.isfinite(H).all(), "non-finite Hessian"
    extra = 0.0
    for _ in range(8):
        try:
            Hd = H if extra == 0 else H + extra * torch.eye(H.shape[0], device=H.device)
            L = torch.linalg.cholesky(Hd.double())
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True).float()
        except torch._C._LinAlgError:
            extra = damp * 10 if extra == 0 else extra * 10
            print(f"  cholesky failed, extra damping {extra:.3e}", flush=True)
    raise RuntimeError("Hessian not PD even with heavy damping")


def rel_err(W0, Q, H):
    D = W0 - Q
    return (torch.einsum("rc,cd,rd->", D, H, D) / torch.einsum("rc,cd,rd->", W0, H, W0)).item()


def refit_alpha(W0, H, idx, q, G, mu_on, row_chunk=None):
    """Exact H-weighted LS for alpha (and mu) with codes fixed. Returns alpha [R,ng,q], mu [R,ng]."""
    R, C = W0.shape
    ng = C // G
    k = q + (1 if mu_on else 0)
    n = ng * k
    if row_chunk is None:
        row_chunk = max(1, int(4e8 // (C * n * 4)))
    S = bcq.codes_to_signs(idx, q)  # [R, C, q]
    if mu_on:
        S = torch.cat([S, torch.ones_like(S[..., :1])], -1)
    alpha = torch.empty(R, ng, q, device=W0.device)
    mu = torch.empty(R, ng, device=W0.device) if mu_on else None
    HW = W0 @ H  # [R, C]
    Hg = H.view(C, ng, G)
    for r0 in range(0, R, row_chunk):
        r1 = min(R, r0 + row_chunk)
        Sg = S[r0:r1].view(r1 - r0, ng, G, k)
        # A_r is block-diagonal over groups: T[r, c, h, p'] = sum_{j' in group h} H[c, j'] S[r, j', p']
        T = torch.einsum("chk,rhkp->rchp", Hg, Sg)
        # M[r, (g,p), (h,p')] = sum_{j in g} S[r, j, p] T[r, j, h, p']
        M = torch.einsum("rgkp,rgkhq->rgphq", Sg, T.view(r1 - r0, ng, G, ng, k)).reshape(r1 - r0, n, n)
        rhs = torch.einsum("rgkp,rgk->rgp", Sg, HW[r0:r1].view(r1 - r0, ng, G)).reshape(r1 - r0, n)
        del T
        diag = M.diagonal(dim1=1, dim2=2)
        M = M + (1e-5 * diag.mean(-1)[:, None, None] + 1e-10) * torch.eye(n, device=W0.device)
        x = torch.linalg.solve(M, rhs[..., None])[..., 0].view(r1 - r0, ng, k)
        alpha[r0:r1] = x[..., :q]
        if mu_on:
            mu[r0:r1] = x[..., q]
    return alpha, mu


def _canon_codes(alpha, mu, idx, q):
    """Make alphas non-negative & sorted desc while keeping the represented weights identical."""
    R, ng, _ = alpha.shape
    G = idx.shape[1] // ng
    S = bcq.codes_to_signs(idx, q).view(R, ng, G, q)
    sgn = torch.where(alpha >= 0, 1.0, -1.0)
    S = S * sgn[:, :, None, :]
    alpha = alpha.abs()
    order = alpha.argsort(-1, descending=True)
    alpha = torch.gather(alpha, -1, order)
    S = torch.gather(S, -1, order[:, :, None, :].expand_as(S))
    bits = ((S > 0).long() << torch.arange(q, device=idx.device)).sum(-1)
    return alpha, mu, bits.view(R, -1)


def dequant(alpha, mu, idx, q, G):
    R, C = idx.shape
    ng = C // G
    S = bcq.codes_to_signs(idx, q).view(R, ng, G, q)
    Wq = (S * alpha[:, :, None, :]).sum(-1)
    if mu is not None:
        Wq = Wq + mu[:, :, None]
    return Wq.view(R, C)


def _uniform_levels(w, q, h=None, grid=20):
    """Asymmetric uniform grid per row with importance-weighted clipping search. w [R, G] -> levels [R, 2^q]."""
    if h is None:
        h = torch.ones_like(w)
    mx, mn = w.amax(-1), w.amin(-1)
    nl = 2 ** q - 1
    best_e = torch.full_like(mx, float("inf"))
    best = None
    for s in torch.linspace(1.0, 0.6, grid):
        lo, hi = mn * s, mx * s
        scale = ((hi - lo) / nl).clamp_min(1e-10)
        k = torch.round((w - lo[:, None]) / scale[:, None]).clamp(0, nl)
        e = (((k * scale[:, None] + lo[:, None]) - w) ** 2 * h).sum(-1)
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        L = lo[:, None].half().float() + scale[:, None].half().float() * torch.arange(nl + 1, device=w.device)
        best = L if best is None else torch.where(better[:, None], L, best)
    return best


@torch.no_grad()
def rtn_uniform(W, H, q, G=64):
    """Round-to-nearest with asymmetric uniform groups (fp16 scale+min per group) — baseline."""
    W0 = W.float()
    R, C = W0.shape
    Lv = _uniform_levels(W0.reshape(-1, G), q)
    k = (W0.reshape(-1, G)[..., None] - Lv[:, None, :]).abs().argmin(-1)
    Q = torch.gather(Lv, 1, k).view(R, C)
    Hd = H.float().clone()
    Hd.diagonal().add_(0.01 * Hd.diagonal().mean())
    return dict(err=rel_err(W0, Q, Hd), Q=Q)


@torch.no_grad()
def quantize_uniform(W, H, q, G=64, blocksize=128, percdamp=0.01, hweight=True, act_order=False):
    """Plain GPTQ with asymmetric uniform groups (fp16 scale+min per group) — baseline.
    act_order permutes whole groups by summed diag(H) (same group-preserving act-order as `quantize`)."""
    W = W.float().clone()
    W0 = W.clone()
    R, C = W.shape
    H = H.float().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)
    hd = H.diagonal().clone()
    perm = None
    if act_order:
        gperm = hd.view(-1, G).sum(-1).argsort(descending=True)
        perm = (gperm[:, None] * G + torch.arange(G, device=W.device)[None]).flatten()
        W, H, hd = W[:, perm], H[perm][:, perm], hd[perm]
    Hinv = robust_hinv(H, damp)
    Q = torch.zeros_like(W)
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % G == 0:
                Lv = _uniform_levels(W1[:, i:i + G], q, hd[c:c + G].expand(R, G) if hweight else None)
            w = W1[:, i]
            k = (w[:, None] - Lv).abs().argmin(-1)
            qv = torch.gather(Lv, 1, k[:, None])[:, 0]
            Q[:, c] = qv
            err = (w - qv) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, perm.argsort()]
        H = H[perm.argsort()][:, perm.argsort()]
    return dict(err=rel_err(W0, Q, H), Q=Q)


@torch.no_grad()
def quantize(W, H, q, G=64, fmt="u8", mu_on=False, SB=256, blocksize=128, percdamp=0.01,
             fit_iters=15, hweight=True, rounds=1, refit=True, act_order=False):
    W0 = W.float().clone()
    R, C = W0.shape
    blocksize = max(blocksize, G)
    assert C % SB == 0 and SB % G == 0 and blocksize % G == 0
    H = H.float().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W0[:, dead] = 0
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)
    hd = H.diagonal().clone()

    perm = None
    if act_order:
        # permute *groups* (keeps groups contiguous so the storage layout is unchanged)
        gscore = hd.view(-1, G).sum(-1)
        gperm = gscore.argsort(descending=True)
        perm = (gperm[:, None] * G + torch.arange(G, device=W.device)[None]).flatten()
    Hp = H[perm][:, perm] if perm is not None else H
    Hinv = robust_hinv(Hp, damp)

    ng = C // G
    # fixed per-row super-block scale for alpha storage, from a plain BCQ fit of W0
    a0, _ = bcq.fit(W0.view(-1, G), q, (hd.expand(R, C).reshape(-1, G) if hweight else None), mu_on, iters=fit_iters)
    a0 = a0.view(R, C // SB, SB // G * q)
    amax = a0.amax(-1) * 1.25  # headroom for compensated weights
    d_sb = {"u8": amax / 255, "pot": amax, "fp16": amax}[fmt].half().float()  # [R, nSB]

    frozen = None  # (alpha, mu) when alternating
    best = None
    for rnd in range(rounds):
        Wp = W0[:, perm] if perm is not None else W0.clone()
        if frozen is None:
            alpha = torch.zeros(R, ng, q, device=W.device)
            mu = torch.zeros(R, ng, device=W.device) if mu_on else None
        else:
            alpha = frozen[0].clone()
            mu = frozen[1].clone() if mu_on else None
        idx = torch.zeros(R, C, dtype=torch.long, device=W.device)
        for i1 in range(0, C, blocksize):
            i2 = min(i1 + blocksize, C)
            W1 = Wp[:, i1:i2].clone()
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                c = i1 + i
                if c % G == 0:
                    g = (perm[c] // G).item() if perm is not None else c // G
                    if frozen is None:
                        h = hd[perm[c:c + G]] if perm is not None else hd[c:c + G]
                        a, m = bcq.fit(W1[:, i:i + G], q, h.expand(R, G) if hweight else None, mu_on, fit_iters)
                        a = _group_alpha_store(a, d_sb[:, g * G // SB], fmt)
                        alpha[:, g] = a
                        if mu_on:
                            mu[:, g] = m
                    cur_a = alpha[:, g]
                    cur_m = mu[:, g] if mu_on else None
                    Lv, _ = bcq.levels(cur_a, cur_m)
                w = W1[:, i]
                k = (w[:, None] - Lv).abs().argmin(-1)
                qv = torch.gather(Lv, 1, k[:, None])[:, 0]
                oc = perm[c] if perm is not None else c
                idx[:, oc] = k
                err = (w - qv) / Hinv1[i, i]
                W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
                Err1[:, i] = err
            Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        if refit:
            alpha, mu = refit_alpha(W0, H, idx, q, G, mu_on)
            alpha, mu, idx = _canon_codes(alpha, mu, idx, q)
        st = _store_all(alpha, mu, fmt, mu_on, SB, G)
        Q = dequant(st["alpha"], st["mu"], idx, q, G)
        e = rel_err(W0, Q, H)
        if best is None or e < best["err"]:
            best = dict(err=e, idx=idx, Q=Q, store=st)
        # next round (if any): GPTQ sweep with these stored alphas frozen
        frozen = (st["alpha"], st["mu"])
    return best


@torch.no_grad()
def quantize_colwise(W, H, q, GR=64, blocksize=256, percdamp=0.01, fit_iters=15, act_order=True, fmt="u8", SB=256):
    """ShiftAddLLM 'acc'-style column-wise BCQ: scales belong to an *input* column and are shared by a
    group of GR consecutive output rows (alpha[g, j, p]).  Mirror image of `quantize`: at the start of
    every GPTQ block the scales of its columns are fitted on the error-compensated weights (one batched
    call), then columns are assigned sequentially with error feedback.
    Storage: u8 codes relative to one fp16 scale per (row group, super-block of SB columns) -> same bpw
    as the row-wise format with G = GR."""
    W0 = W.float().clone()
    R, C = W0.shape
    assert R % GR == 0 and C % SB == 0 and blocksize % SB == 0
    H = H.float().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W0[:, dead] = 0
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)
    hd = H.diagonal().clone()
    if act_order:  # permute super-blocks of columns (keeps the d-sharing layout intact)
        gscore = hd.view(-1, SB).sum(-1)
        perm = (gscore.argsort(descending=True)[:, None] * SB + torch.arange(SB, device=W.device)[None]).flatten()
    else:
        perm = torch.arange(C, device=W.device)
    Hinv = robust_hinv(H[perm][:, perm], damp)
    Wp = W0[:, perm].clone()
    Q = torch.zeros_like(Wp)
    ng = R // GR
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        nb = i2 - i1
        W1 = Wp[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        # batched column fits: vectors (group g, column j) of length GR -> alpha [ng, nb, q]
        wv = W1.view(ng, GR, nb).permute(0, 2, 1).reshape(-1, GR)
        a, _ = bcq.fit(wv, q, None, False, fit_iters)
        a = a.view(ng, nb, q)
        if fmt == "u8":  # one fp16 d per (row group, super-block of SB columns)
            av = a.view(ng, nb // SB, SB * q)
            d = (av.amax(-1, keepdim=True) / 255).clamp_min(1e-12).half().float()
            a = (torch.round(av / d).clamp(0, 255) * d).view(ng, nb, q)
        for i in range(nb):
            w = W1[:, i]
            _, qv = bcq.assign(w.view(ng, GR), a[:, i])  # nearest level per group
            qv = qv.reshape(-1)
            Q[:, i1 + i] = qv
            err = (w - qv) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Q = Q[:, perm.argsort()]
    return dict(err=rel_err(W0, Q, H), Q=Q)
