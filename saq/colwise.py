"""Column-wise ("acc"-mode) ShiftAdd format: scales belong to the INPUT column and are shared by a
row group of GR=64 output rows; a per-row, per-super-block fp16 gain d absorbs the row magnitude.

    w[r, j] = d[r, b] * sum_p  A[g, b, p, jj] * s_p(r, j)      (g = r // GR, b = j // 256, jj = j % 256)

A is stored as u8 codes (0..255).  This module holds the fit (GPTQ sweep with column-wise BCQ scales),
the exact H-weighted refit of (A, d) with the signs frozen, and the storage packing used by the
`sac2/sac3/sac4` ggml types (see llama.cpp/shiftadd-b11538.patch).
"""
import torch

from . import bcq
from .gptq_bcq import rel_err, robust_hinv

GR, SB = 64, 256


@torch.no_grad()
def fit(W, H, q, blocksize=256, percdamp=0.01, fit_iters=15, act_order=True, refit_rounds=2):
    """-> dict(err, Q, idx [R,C] uint8 level codes, A [ng,nB,q,SB] dequantized scales, d [R,nB])."""
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
    dev = W.device
    if act_order:  # permute whole super-blocks so the d-sharing layout survives
        gscore = hd.view(-1, SB).sum(-1)
        perm = (gscore.argsort(descending=True)[:, None] * SB + torch.arange(SB, device=dev)[None]).flatten()
    else:
        perm = torch.arange(C, device=dev)
    Hinv = robust_hinv(H[perm][:, perm], damp)
    Wp = W0[:, perm].clone()
    ng = R // GR
    idx = torch.zeros(R, C, dtype=torch.long, device=dev)
    alpha = torch.zeros(ng, C, q, device=dev)  # permuted column order
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        nb = i2 - i1
        W1 = Wp[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        wv = W1.view(ng, GR, nb).permute(0, 2, 1).reshape(-1, GR)
        a, _ = bcq.fit(wv, q, None, False, fit_iters)
        a = a.view(ng, nb, q)
        av = a.view(ng, nb // SB, SB * q)  # snap to the u8 grid of each (group, super-block)
        dd = (av.amax(-1, keepdim=True) / 255).clamp_min(1e-12)
        a = (torch.round(av / dd).clamp(0, 255) * dd).view(ng, nb, q)
        alpha[:, i1:i2] = a
        for i in range(nb):
            w = W1[:, i]
            k, qv = bcq.assign(w.view(ng, GR), a[:, i])
            idx[:, i1 + i] = k.reshape(-1)
            qv = qv.reshape(-1)
            err = (w - qv) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    inv = perm.argsort()
    idx = idx[:, inv]
    alpha = alpha[:, inv]
    # initial (A, d): d = 1 per row, A = alpha
    A = alpha.view(ng, C // SB, SB, q).permute(0, 1, 3, 2).contiguous()  # [ng, nB, q, SB]
    d = torch.ones(R, C // SB, device=dev)
    for _ in range(refit_rounds):
        A, d = refit(W0, H, idx, A, d, q)
    A, d, idx = canonical(A, d, idx, q)
    Q = dequant(idx, A, d, q)
    return dict(err=rel_err(W0, Q, H), Q=Q, idx=idx.to(torch.uint8), A=A, d=d)


def signs(idx, q):
    return bcq.codes_to_signs(idx.long(), q)  # [R, C, q] in {-1,+1}


@torch.no_grad()
def dequant(idx, A, d, q):
    R, C = idx.shape
    ng, nB = A.shape[:2]
    S = signs(idx, q).view(R, nB, SB, q)
    Ar = A.permute(0, 1, 3, 2).repeat_interleave(GR, 0)  # [R, nB, SB, q]
    return ((S * Ar).sum(-1) * d[:, :, None]).reshape(R, C)


@torch.no_grad()
def refit(W0, H, idx, A, d, q, rounds=2):
    """Alternating exact LS on the H-weighted output error with signs frozen:
    (1) per row group, solve the C*q table entries jointly (all super-blocks couple through H);
    (2) per row, solve the nB gains d[r, :] jointly."""
    R, C = W0.shape
    ng, nB = A.shape[:2]
    S = signs(idx, q)  # [R, C, q]
    WH = W0 @ H  # [R, C]
    Hq = H[:, None, :, None]  # broadcast helper for the (j,p),(j',p') expansion
    for _ in range(rounds):
        # ---- (1) table entries per group: M c = rhs,  M = (Sd^T Sd) * (H kron 1)
        newA = torch.empty_like(A)
        for g in range(ng):
            rs = slice(g * GR, (g + 1) * GR)
            Sd = (S[rs] * d[rs].repeat_interleave(SB, 1)[:, :, None]).reshape(GR, C * q)  # [GR, Cq]
            M = (Sd.T @ Sd).view(C, q, C, q) * Hq
            M = M.reshape(C * q, C * q)
            M.diagonal().add_(1e-6 * M.diagonal().mean() + 1e-10)
            rhs = (Sd.view(GR, C, q) * (WH[rs])[:, :, None]).sum(0).reshape(C * q)
            c = torch.linalg.solve(M, rhs[:, None])[:, 0].view(C, q)
            newA[g] = c.view(nB, SB, q).permute(0, 2, 1)
            del M
        A = newA.clamp_min(0)  # negative table entries are folded into the signs by canonical()
        # ---- (2) per-row gains: nB-dim LS per row (blocks couple through H)
        Ar = A.permute(0, 1, 3, 2).repeat_interleave(GR, 0)  # [R, nB, SB, q]
        V = (S.view(R, nB, SB, q) * Ar).sum(-1)  # [R, nB, SB]  (v_{r,b} restricted to block b)
        Vf = torch.zeros(R, nB, C, device=W0.device)
        ar = torch.arange(nB, device=W0.device)
        Vf.view(R, nB, nB, SB)[:, ar, ar] = V
        G = torch.einsum("rbc,cd,red->rbe", Vf, H, Vf)  # [R, nB, nB]
        G.diagonal(dim1=1, dim2=2).add_(1e-6 * G.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True) + 1e-10)
        rhs = torch.einsum("rbc,rc->rb", Vf, WH)
        d = torch.linalg.solve(G, rhs[..., None])[..., 0]
    return A, d


@torch.no_grad()
def canonical(A, d, idx, q):
    """Make d >= 0 and the stored table u8-valid: fold sign(d) into the signs, snap A to the u8 grid of
    each (g, b) with the normaliser folded into d[r, b]."""
    R, C = idx.shape
    ng, nB = A.shape[:2]
    S = signs(idx, q).view(R, nB, SB, q)
    sd = torch.where(d >= 0, 1.0, -1.0)  # [R, nB]
    S = S * sd[:, :, None, None]
    d = d.abs()
    dd = A.flatten(2).amax(-1).clamp_min(1e-12) / 255  # [ng, nB]
    codes = torch.round(A / dd[:, :, None, None]).clamp(0, 255)
    A = codes * dd[:, :, None, None]
    d = (d * dd.repeat_interleave(GR, 0)).half().float()
    A = A / dd[:, :, None, None]  # table now holds the integer codes; d carries the scale
    bits = ((S > 0).long() << torch.arange(q, device=idx.device)).sum(-1)
    return A, d, bits.view(R, C)


# ------------------------------------------------------------------ storage packing (sacQ blocks)

def pack_blocks(idx, A, d, q):
    """idx [R,C] uint8, A [ng,nB,q,SB] integer codes (0..255), d [R,nB] -> bytes [R, nB*(q*36+3)]."""
    R, C = idx.shape
    nB = C // SB
    ng = R // GR
    idx = idx.view(R, nB, SB).to(torch.int32)
    planes = []
    for p in range(q):
        bits = ((idx >> p) & 1).to(torch.uint8).view(R, nB, 32, 8)
        planes.append((bits << torch.arange(8, dtype=torch.uint8)).sum(-1, dtype=torch.uint8))
    qs = torch.cat(planes, -1)  # [R, nB, q*32]
    T = A.to(torch.uint8).reshape(ng, nB, q * SB)  # flattened table T[p*256 + jj]
    slices = T.view(ng, nB, GR, q * 4).permute(0, 2, 1, 3).reshape(R, nB, q * 4)  # row t gets bytes [t*q*4, +q*4)
    db = d.to(torch.float16).view(torch.uint8).view(R, nB, 2)
    t = (torch.arange(R) % GR).to(torch.uint8)[:, None, None].expand(R, nB, 1)
    return torch.cat([qs, slices, db, t], -1).reshape(R, -1).numpy()


def dequant_blocks(blk, q, C):
    """numpy reference dequant of the packed layout (self-check for the exporter)."""
    R = blk.shape[0]
    nB = C // SB
    bs = q * 36 + 3
    b = torch.from_numpy(blk.reshape(R, nB, bs).copy())
    qs = b[..., :q * 32].reshape(R, nB, q, 32)
    sl = b[..., q * 32:q * 32 + q * 4]
    d = b[..., q * 36:q * 36 + 2].contiguous().view(torch.float16).float().view(R, nB)
    t = b[..., q * 36 + 2].long()
    assert torch.equal(t, (torch.arange(R) % GR)[:, None].expand(R, nB))
    T = sl.view(R // GR, GR, nB, q * 4).permute(0, 2, 1, 3).reshape(R // GR, nB, q, SB).float()
    bits = (qs[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1
    S = bits.reshape(R, nB, q, SB).float() * 2 - 1
    Tr = T.repeat_interleave(GR, 0)
    return ((S * Tr).sum(2) * d[:, :, None]).reshape(R, C)


# ------------------------------------------------------------------ generalised block-wise (bc x br) tiles

@torch.no_grad()
def fit_block(W, H, q, bc=8, br=32, blocksize=256, percdamp=0.01, fit_iters=15, act_order=True, refit_rounds=2):
    """One scale per plane per tile of `bc` input columns x `br` output rows (ShiftAddLLM Fig. 3(a):
    bc=8, br=R/8).  bc=256, br=1 is the row-wise SA*_256 format; bc=1, br=64 is column-wise.
    GPTQ sweep: at every bc-column boundary the tile scales are fitted (batched over row groups) on
    the error-compensated weights, then the bc columns are assigned sequentially with error feedback.
    -> dict(err, Q, idx, T [ng, nK, q] dequantized scales (nK = C/bc), d [R, nB])."""
    W0 = W.float().clone()
    R, C = W0.shape
    assert R % br == 0 and C % SB == 0 and SB % bc == 0 and blocksize % SB == 0
    H = H.float().clone()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    W0[:, dead] = 0
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)
    hd = H.diagonal().clone()
    dev = W.device
    if act_order:
        gscore = hd.view(-1, SB).sum(-1)
        perm = (gscore.argsort(descending=True)[:, None] * SB + torch.arange(SB, device=dev)[None]).flatten()
    else:
        perm = torch.arange(C, device=dev)
    Hinv = robust_hinv(H[perm][:, perm], damp)
    Wp = W0[:, perm].clone()
    ng, nK = R // br, C // bc
    idx = torch.zeros(R, C, dtype=torch.long, device=dev)
    T = torch.zeros(ng, nK, q, device=dev)
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = Wp[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % bc == 0:  # fit this column block's tiles on the compensated weights
                tile = W1[:, i:i + bc].reshape(ng, br, bc).reshape(ng, br * bc)
                a, _ = bcq.fit(tile, q, None, False, fit_iters)  # [ng, q]
                T[:, c // bc] = a
                Lv, _ = bcq.levels(a)  # [ng, 2^q]
            w = W1[:, i].view(ng, br)
            k = (w[..., None] - Lv[:, None, :]).abs().argmin(-1)
            qv = torch.gather(Lv, 1, k)
            idx[:, c] = k.reshape(-1)
            err = (W1[:, i] - qv.reshape(-1)) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    inv = perm.argsort()
    idx = idx[:, inv]
    T = T.view(ng, nK, q)[:, (inv.view(-1, bc)[:, 0] // bc)]  # super-block permutation keeps tiles intact
    d = torch.ones(R, C // SB, device=dev)
    for _ in range(refit_rounds):
        T, d = refit_block(W0, H, idx, T, d, q, bc, br)
    T, d, idx = canonical_block(T, d, idx, q, bc, br)
    Q = dequant_block(idx, T, d, q, bc, br)
    return dict(err=rel_err(W0, Q, H), Q=Q, idx=idx.to(torch.uint8), T=T, d=d)


def _expand_T(T, bc, br, R, C):
    """[ng, nK, q] -> per element [R, C, q]"""
    return T.repeat_interleave(br, 0).repeat_interleave(bc, 1)


@torch.no_grad()
def dequant_block(idx, T, d, q, bc, br):
    R, C = idx.shape
    S = signs(idx, q)
    w = (S * _expand_T(T, bc, br, R, C)).sum(-1)
    return w * d.repeat_interleave(SB, 1)


@torch.no_grad()
def refit_block(W0, H, idx, T, d, q, bc, br, rounds=2):
    """Exact alternating LS with signs frozen. (1) per row group: all (column block, plane) scales
    jointly:  M = P^T ((Sd^T Sd) * (H kron 1)) P  with P the column-block summation;  (2) per row:
    the nB gains jointly."""
    R, C = W0.shape
    ng, nK = T.shape[:2]
    nB = C // SB
    S = signs(idx, q)  # [R, C, q]
    WH = W0 @ H
    Hq = H[:, None, :, None]
    for _ in range(rounds):
        newT = torch.empty_like(T)
        for g in range(ng):
            rs = slice(g * br, (g + 1) * br)
            Sd = (S[rs] * d[rs].repeat_interleave(SB, 1)[:, :, None]).reshape(br, C * q)
            G = (Sd.T @ Sd).view(C, q, C, q) * Hq  # [(j,p),(j',p')] with H
            M = G.view(nK, bc, q, nK, bc, q).sum((1, 4)).reshape(nK * q, nK * q)
            del G
            M.diagonal().add_(1e-6 * M.diagonal().mean() + 1e-10)
            rhs = (Sd.view(br, C, q) * WH[rs][:, :, None]).sum(0).view(nK, bc, q).sum(1).reshape(-1)
            newT[g] = torch.linalg.solve(M, rhs[:, None])[:, 0].view(nK, q)
        T = newT.clamp_min(0)
        V = (S * _expand_T(T, bc, br, R, C)).sum(-1)  # [R, C] unscaled rows
        Vf = torch.zeros(R, nB, C, device=W0.device)
        ar = torch.arange(nB, device=W0.device)
        Vf.view(R, nB, nB, SB)[:, ar, ar] = V.view(R, nB, SB)
        Gd = torch.einsum("rbc,cd,red->rbe", Vf, H, Vf)
        Gd.diagonal(dim1=1, dim2=2).add_(1e-6 * Gd.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True) + 1e-10)
        rhs = torch.einsum("rbc,rc->rb", Vf, WH)
        d = torch.linalg.solve(Gd, rhs[..., None])[..., 0]
    return T, d


@torch.no_grad()
def canonical_block(T, d, idx, q, bc, br):
    """d >= 0 (sign folded into the codes); table snapped to u8 per (row group, super-block) with the
    normaliser folded into d[r, b]."""
    R, C = idx.shape
    ng, nK = T.shape[:2]
    nB = C // SB
    kps = SB // bc
    S = signs(idx, q).view(R, nB, SB, q)
    sd = torch.where(d >= 0, 1.0, -1.0)
    S = S * sd[:, :, None, None]
    d = d.abs()
    Tb = T.view(ng, nB, kps, q)
    dd = Tb.flatten(2).amax(-1).clamp_min(1e-12) / 255  # [ng, nB]
    codes = torch.round(Tb / dd[:, :, None, None]).clamp(0, 255)
    d = (d * dd.repeat_interleave(br, 0)).half().float()
    bits = ((S > 0).long() << torch.arange(q, device=idx.device)).sum(-1)
    return codes.view(ng, nK, q), d, bits.view(R, C)


def bpw_block(q, bc, br):
    """bits/weight of the sab layout: signs + table slice + fp16 d + slot byte per 256 weights."""
    L = q * (SB // bc) / br
    return (q * 32 + L + 3) * 8 / SB


# ------------------------------------------------------------------ storage packing (sab<Q>_<bc>x<br> blocks)

def pack_block(idx, T, d, q, bc, br):
    """idx [R,C] uint8 level codes, T [ng,nK,q] integer codes 0..255, d [R,nB] -> bytes [R, nB*(q*32+L+3)]
    block = qs[q*32] | a[L] (this row's slice of the (g,b) table T[p*(256/bc)+k]) | d fp16 | t=r%br."""
    R, C = idx.shape
    nB, kps = C // SB, SB // bc
    ng = R // br
    L = q * kps // br
    assert L * br == q * kps, "table slice must be whole bytes: q*(256/bc) % br == 0"
    idx = idx.view(R, nB, SB).to(torch.int32)
    planes = []
    for p in range(q):
        bits = ((idx >> p) & 1).to(torch.uint8).view(R, nB, 32, 8)
        planes.append((bits << torch.arange(8, dtype=torch.uint8)).sum(-1, dtype=torch.uint8))
    qs = torch.cat(planes, -1)  # [R, nB, q*32]
    tab = T.to(torch.uint8).view(ng, nB, kps, q).permute(0, 1, 3, 2).reshape(ng, nB, q * kps)  # T[p*kps + k]
    slices = tab.view(ng, nB, br, L).permute(0, 2, 1, 3).reshape(R, nB, L)
    db = d.to(torch.float16).view(torch.uint8).view(R, nB, 2)
    t = (torch.arange(R) % br).to(torch.uint8)[:, None, None].expand(R, nB, 1)
    return torch.cat([qs, slices, db, t], -1).reshape(R, -1).numpy()


def dequant_block_bytes(blk, q, bc, br, C):
    """numpy/torch reference dequant of the packed sab layout."""
    R = blk.shape[0]
    nB, kps = C // SB, SB // bc
    L = q * kps // br
    bs = q * 32 + L + 3
    b = torch.from_numpy(blk.reshape(R, nB, bs).copy())
    qs = b[..., :q * 32].reshape(R, nB, q, 32)
    sl = b[..., q * 32:q * 32 + L]
    d = b[..., q * 32 + L:q * 32 + L + 2].contiguous().view(torch.float16).float().view(R, nB)
    t = b[..., q * 32 + L + 2].long()
    assert torch.equal(t, (torch.arange(R) % br)[:, None].expand(R, nB)), "slot bytes"
    tab = sl.view(R // br, br, nB, L).permute(0, 2, 1, 3).reshape(R // br, nB, q, kps).float()  # [ng,nB,q,kps]
    bits = (qs[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1
    S = bits.reshape(R, nB, q, SB).float() * 2 - 1
    Te = tab.repeat_interleave(br, 0).repeat_interleave(bc, 3)  # [R, nB, q, SB]
    return ((S * Te).sum(2) * d[:, :, None]).reshape(R, C)
