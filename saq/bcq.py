"""Binary-coding quantization (BCQ) primitives — the ShiftAddLLM weight form.

A group of G weights w (along the input dim of one output row) is represented as

    w ~= mu + sum_p alpha_p * b_p,   b_p in {-1,+1}^G,  alpha_1 >= ... >= alpha_q >= 0

so a mat-vec becomes sign-adds of activations per bit-plane followed by q scalings per
group (shifts when alpha is a power of two).  mu (optional) is a per-group offset.

All functions are batched over leading dims: w is [..., G], importance h is [..., G].
"""
import itertools

import torch


def sign_table(q, device):
    """All 2^q sign patterns, [2^q, q] in {-1,+1}; row k has bit p = (k >> p) & 1 -> +1."""
    k = torch.arange(2 ** q, device=device)
    bits = (k[:, None] >> torch.arange(q, device=device)[None, :]) & 1
    return bits.float() * 2 - 1


def levels(alpha, mu=None):
    """alpha [..., q] -> representable values [..., 2^q] (and sign table)."""
    S = sign_table(alpha.shape[-1], alpha.device)
    L = alpha @ S.T
    if mu is not None:
        L = L + mu[..., None]
    return L, S


def assign(w, alpha, mu=None):
    """Nearest level for every weight. w [..., G], alpha [..., q] -> code idx [..., G], qw [..., G]."""
    L, _ = levels(alpha, mu)
    idx = (w[..., :, None] - L[..., None, :]).abs().argmin(-1)
    qw = torch.gather(L, -1, idx)
    return idx, qw


def codes_to_signs(idx, q):
    """code idx [...] -> signs [..., q]"""
    p = torch.arange(q, device=idx.device)
    return ((idx[..., None] >> p) & 1).float() * 2 - 1


def solve_alpha(w, B, h, mu_on):
    """Weighted LS for alpha (and mu) with fixed signs. w [N,G], B [N,G,q], h [N,G]."""
    if mu_on:
        B = torch.cat([B, torch.ones_like(B[..., :1])], -1)
    Bh = B * h[..., None]
    A = Bh.transpose(-1, -2) @ B  # [N, q, q]
    r = (Bh * w[..., None]).sum(-2)  # [N, q]
    n = A.shape[-1]
    A = A + 1e-6 * A.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(n, device=w.device) \
        + 1e-12 * torch.eye(n, device=w.device)
    x = torch.linalg.solve(A, r[..., None])[..., 0]
    if mu_on:
        return x[..., :-1], x[..., -1]
    return x, None


def fit(w, q, h=None, mu_on=False, iters=15, max_elems=2 ** 27):
    """Fit BCQ to groups. w [N, G] (float32), h importance [N, G] or None.

    Row-chunked so the exhaustive level search ([N, G, 2^q]) stays within ~max_elems floats.
    """
    N, G = w.shape
    step = max(1, max_elems // (G * 2 ** q))
    if N > step:
        parts = [fit(w[i:i + step], q, None if h is None else h[i:i + step], mu_on, iters, max_elems)
                 for i in range(0, N, step)]
        alpha = torch.cat([a for a, _ in parts])
        mu = torch.cat([m for _, m in parts]) if mu_on else None
        return alpha, mu
    return _fit(w, q, h, mu_on, iters)


def _fit(w, q, h=None, mu_on=False, iters=15):
    """Unchunked BCQ fit (see fit).

    Greedy residual init then alternating (alpha, mu) LS / exhaustive code assignment.
    Returns alpha [N, q] (sorted desc, >= 0), mu [N] or None.
    """
    if h is None:
        h = torch.ones_like(w)
    h = h.clamp_min(1e-12)
    mu = (w * h).sum(-1) / h.sum(-1) if mu_on else None
    r = w - mu[:, None] if mu_on else w.clone()
    alphas = []
    for _ in range(q):
        b = torch.where(r >= 0, 1.0, -1.0)
        a = (r.abs() * h).sum(-1) / h.sum(-1)
        alphas.append(a)
        r = r - a[:, None] * b
    alpha = torch.stack(alphas, -1)
    best_alpha, best_mu = alpha, mu
    best_err = _err(w, alpha, mu, h)
    for _ in range(iters):
        idx, _ = assign(w, alpha, mu)
        B = codes_to_signs(idx, q)
        alpha, mu = solve_alpha(w, B, h, mu_on)
        alpha, mu = canonical(alpha, mu)
        err = _err(w, alpha, mu, h)
        better = err < best_err
        best_alpha = torch.where(better[:, None], alpha, best_alpha)
        if mu_on:
            best_mu = torch.where(better, mu, best_mu)
        best_err = torch.minimum(err, best_err)
    return best_alpha, best_mu


def canonical(alpha, mu=None):
    """Sign of each alpha can be folded into its bit-plane: make alpha >= 0, sort descending."""
    alpha = alpha.abs().sort(-1, descending=True).values
    return alpha, mu


def _err(w, alpha, mu, h):
    _, qw = assign(w, alpha, mu)
    return ((w - qw) ** 2 * h).sum(-1)


# ---------------------------------------------------------------- alpha storage formats
#
# Alphas of a super-block (SB weights along the input dim, SB/G groups x q planes) are stored
# relative to one fp16 scale d:
#   "u8"  : alpha = d * code,           code in [0, 255]
#   "pot" : alpha = d * 2^-e,           e in [0, 15]  (pure shift; ShiftAddLLM-style)
#   "fp16": alpha kept in fp16 (reference)
# mu (if used) is int8:  mu = dm * code, code in [-127, 127], one fp16 dm per super-block.

def quant_alpha(alpha, fmt):
    """alpha [R, nSB, ng, q] -> dequantized alpha (same shape), plus storage dict."""
    if fmt == "fp16":
        return alpha.half().float(), {}
    amax = alpha.flatten(-2).amax(-1).clamp_min(1e-12)  # [R, nSB]
    if fmt == "u8":
        d = (amax / 255).half().float()
        code = torch.round(alpha / d[..., None, None]).clamp(0, 255)
        return code * d[..., None, None], {"d": d, "code": code.to(torch.uint8)}
    if fmt == "pot":
        d = amax.half().float()
        e = torch.round(-torch.log2((alpha / d[..., None, None]).clamp_min(2 ** -15))).clamp(0, 15)
        return d[..., None, None] * torch.exp2(-e), {"d": d, "e": e.to(torch.uint8)}
    raise ValueError(fmt)


def quant_mu(mu):
    """mu [R, nSB, ng] -> dequantized, storage."""
    amax = mu.abs().amax(-1).clamp_min(1e-12)
    d = (amax / 127).half().float()
    code = torch.round(mu / d[..., None]).clamp(-127, 127)
    return code * d[..., None], {"dm": d, "mcode": code.to(torch.int8)}


def bits_per_weight(q, G, fmt, mu_on, SB=256):
    alpha_bits = {"u8": 8, "pot": 4, "fp16": 16}[fmt]
    b = q + q * alpha_bits / G + (16 / SB if fmt != "fp16" else 0)
    if mu_on:
        b += 8 / G + 16 / SB
    return b
