"""Trainable BCQ linear: frozen binary codes, trainable per-group scales (alpha, optionally mu).

W[r, j] = sum_p alpha[r, g(j), p] * s_p(idx[r, j])  (+ mu[r, g(j)])
Forward rounds alpha to its storage grid (u8 codes x per-super-block fp16 scale) with a
straight-through estimator, so what is trained is exactly what ships in the GGUF.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bcq


class BlockBCQLinear(nn.Module):
    """Block-wise (bc cols x br rows) ShiftAdd linear for distillation: frozen signs, trainable table
    codes (relative) and per-(row, super-block) gains.  Forward snaps codes to u8 and d to fp16 (STE)."""

    def __init__(self, idx, T, d, q, bc, br, col_perm=None):
        super().__init__()
        from . import colwise
        self.cw = colwise
        R, C = idx.shape
        self.q, self.bc, self.br = q, bc, br
        self.out_features, self.in_features = R, C
        self.register_buffer("idx", idx.to(torch.uint8).contiguous(), persistent=False)
        self.register_buffer("T0", T.float().clone(), persistent=False)  # [ng, nK, q] codes
        self.register_buffer("d0", d.float().clone(), persistent=False)  # [R, nB]
        self.thetaT = nn.Parameter(torch.zeros_like(self.T0))
        self.thetad = nn.Parameter(torch.zeros_like(self.d0))
        self.register_buffer("inv_perm", col_perm.argsort().to(idx.device) if col_perm is not None else None,
                             persistent=False)
        self.snap = True

    def stored(self):
        T = self.T0 * (1 + self.thetaT)
        d = self.d0 * (1 + self.thetad)
        if self.snap:
            Tq = torch.round(T).clamp(0, 255)
            T = T + (Tq - T).detach()
            d = d + (d.half().float() - d).detach()
        return T, d

    def weight(self):
        T, d = self.stored()
        W = self.cw.dequant_block(self.idx.long(), T, d, self.q, self.bc, self.br)
        if self.inv_perm is not None:
            W = W[:, self.inv_perm]
        return W

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype))

    @torch.no_grad()
    def export(self):
        T, d = self.stored()
        return torch.round(T).clamp(0, 255).to(torch.uint8), d.half().float()


class BCQLinear(nn.Module):
    def __init__(self, idx, alpha, q, G, mu=None, col_perm=None, fmt="u8", SB=256, gran="group"):
        super().__init__()
        R, C = idx.shape
        self.q, self.G, self.fmt, self.SB = q, G, fmt, SB
        self.out_features, self.in_features = R, C
        self.register_buffer("idx", idx.to(torch.uint8).contiguous(), persistent=False)
        # trained relatively: alpha = alpha0 * (1 + theta), so lr is a fractional step size
        self.register_buffer("alpha0", alpha.float().clone(), persistent=False)  # [R, ng, q]
        # gran: "group" = one theta per (row, group, plane); "row" = per (row, plane), shared over groups
        tshape = self.alpha0.shape if gran == "group" else (R, 1, q)
        self.theta = nn.Parameter(torch.zeros(tshape, device=idx.device))
        self.snap = True  # round to the u8 storage grid in forward (off while training)
        self.mu = nn.Parameter(mu.float().clone()) if mu is not None else None
        self.register_buffer("sign", bcq.sign_table(q, idx.device), persistent=False)  # [2^q, q]
        self.register_buffer("inv_perm", col_perm.argsort().to(idx.device) if col_perm is not None else None,
                             persistent=False)

    @property
    def alpha(self):
        return self.alpha0 * (1 + self.theta)

    def stored_alpha(self):
        """alpha snapped to the u8 storage grid (STE)."""
        a = self.alpha
        if self.fmt != "u8" or not self.snap:
            return a
        R, ng, q = a.shape
        gps = self.SB // self.G
        av = a.view(R, ng // gps, gps * q)
        d = (av.detach().abs().amax(-1, keepdim=True) / 255).clamp_min(1e-12)
        d = d.half().float()
        aq = torch.round(av / d).clamp(0, 255) * d
        return (av + (aq - av).detach()).view(R, ng, q)

    def weight(self):
        a = self.stored_alpha()
        lv = a @ self.sign.T  # [R, ng, 2^q]
        if self.mu is not None:
            lv = lv + self.mu[..., None]
        R, ng, _ = lv.shape
        W = torch.gather(lv, 2, self.idx.view(R, ng, self.G).long()).view(R, -1)
        if self.inv_perm is not None:
            W = W[:, self.inv_perm]
        return W

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype))

    @torch.no_grad()
    def export_alpha(self):
        """-> (dequantized alpha, store dict) in the same format the quantizer saves."""
        a = self.alpha.detach().abs()
        # sorting/abs keeps semantics only if codes are re-labelled; keep raw if any alpha < 0
        R, ng, q = a.shape
        gps = self.SB // self.G
        av = self.alpha.detach().view(R, ng // gps, gps, q)
        aq, st = bcq.quant_alpha(av.clamp_min(0), self.fmt)
        return aq.view(R, ng, q), st
