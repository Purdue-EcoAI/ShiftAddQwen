"""Does a residual-stream Hadamard rotation (foldable offline) reduce BCQ error? Per-matrix check.

Readers of the residual (qkv, z, q/k/v, gate, up): W' = W diag(1+g) Q,  H' = Q^T diag(1+g)^-1 H diag(1+g)^-1 Q
Writers to the residual (out_proj, o_proj, down): W' = Q^T W, H unchanged.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from . import gptq_bcq
from .model import MODEL, WORK

torch.backends.cuda.matmul.allow_tf32 = False
LAB = WORK / "data" / "lab"


def hadamard(n, seed=0):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (H * s[None, :]) / n ** 0.5  # random-sign Hadamard, orthogonal


def norm_weight(name):
    import json as _j
    idx = _j.load(open(Path(MODEL) / "model.safetensors.index.json"))["weight_map"]
    f = Path(MODEL) / idx[name]
    with safe_open(str(f), "pt") as fh:
        return fh.get_tensor(name).float()


READ = ("in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
WRITE = ("out_proj", "o_proj", "down_proj")


def main(files, qs=(2, 3)):
    Q = hadamard(4096).cuda()
    for fn in files:
        d = torch.load(LAB / fn)
        W, H = d["W"].cuda().float(), d["H"].cuda().float()
        li = int(fn.split(".")[0][1:])
        sub = fn.split(".")[2]
        if sub in READ:
            ln = "input_layernorm" if sub not in ("gate_proj", "up_proj") else "post_attention_layernorm"
            g = 1 + norm_weight(f"model.language_model.layers.{li}.{ln}.weight").cuda()
            Wr = (W * g[None]) @ Q
            Hr = Q.T @ (H / g[:, None] / g[None, :]) @ Q
        else:
            Wr, Hr = Q.T @ W, H
        for q in qs:
            for G in (64,):
                e0 = gptq_bcq.quantize(W, H, q, G=G)["err"]
                e1 = gptq_bcq.quantize(Wr, Hr, q, G=G)["err"]
                u0 = gptq_bcq.quantize_uniform(W, H, q, G=G)["err"]
                u1 = gptq_bcq.quantize_uniform(Wr, Hr, q, G=G)["err"]
                print(json.dumps(dict(file=fn, q=q, G=G, bcq=e0, bcq_rot=e1, unif=u0, unif_rot=u1)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
