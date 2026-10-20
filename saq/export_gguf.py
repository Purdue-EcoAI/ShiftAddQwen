"""Write a GGUF whose decoder linears are ShiftAdd/BCQ tensors (ggml types SA2/SA3/SA4).

  python -m saq.export_gguf --run work/runs/X --base base.gguf --ref BF16.gguf --out X.gguf

--base : GGUF produced by stock llama-quantize from the BF16 GGUF (decides the types of all
         non-BCQ tensors: embeddings, lm head, DeltaNet gates, MTP block, norms ...)
--ref  : the BF16 GGUF, used to verify the HF->GGUF name/row/column mapping bit-exactly.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# gguf-py from the patched llama.cpp (llama.cpp/shiftadd-b11538.patch adds the SA types)
LLAMA = Path(os.environ.get("LLAMA_CPP", Path(__file__).resolve().parents[1] / "work/llama.cpp")) / "gguf-py"
sys.path.insert(0, str(LLAMA))
import gguf  # noqa: E402  (fork with SA types)

from . import colwise, gptq_bcq  # noqa: E402

T = gguf.GGMLQuantizationType
SA_TYPE = {(2, 64): T.SA2, (3, 64): T.SA3, (4, 64): T.SA4, (2, 32): T.SA2S,
           (2, 128): T.SA2_128, (3, 128): T.SA3_128, (4, 128): T.SA4_128,
           (2, 256): T.SA2_256, (3, 256): T.SA3_256, (4, 256): T.SA4_256,
           (5, 64): T.SA5, (5, 128): T.SA5_128, (5, 256): T.SA5_256,
           (6, 64): T.SA6, (6, 128): T.SA6_128, (6, 256): T.SA6_256}

HF2GGUF = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
NK, NVPK, HKD, HVD = 16, 2, 128, 128  # Qwen3.5-9B linear-attention geometry


def v_reorder_idx(n_heads_k, per_k, head_dim):
    n = n_heads_k * per_k * head_dim
    return torch.arange(n).reshape(n_heads_k, per_k, head_dim).permute(1, 0, 2).reshape(n)


def row_perm(sub, R):
    """GGUF row r holds HF row perm[r]."""
    if sub == "linear_attn.in_proj_qkv":
        qk = 2 * NK * HKD
        return torch.cat([torch.arange(qk), qk + v_reorder_idx(NK, NVPK, HVD)])
    if sub == "linear_attn.in_proj_z":
        return v_reorder_idx(NK, NVPK, HVD)
    return torch.arange(R)


def col_perm(sub, C):
    if sub == "linear_attn.out_proj":
        return v_reorder_idx(NK, NVPK, HVD)
    return torch.arange(C)


def pack_blocks(idx, code, d, q, G):
    """idx [R, C] uint8 level codes, code [R, nSB, gps, q] uint8 alpha codes, d [R, nSB] fp16
    -> bytes [R, nSB * blocksize] in block_saQ layout {qs[q*32], a[q*256/G], d}."""
    R, C = idx.shape
    nSB = C // 256
    idx = idx.view(R, nSB, 256).to(torch.int32)
    planes = []
    for p in range(q):
        bits = ((idx >> p) & 1).to(torch.uint8).view(R, nSB, 32, 8)
        byte = (bits << torch.arange(8, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)  # bit j%8 of byte j/8
        planes.append(byte)
    qs = torch.cat(planes, -1)  # [R, nSB, q*32]
    a = code.reshape(R, nSB, (256 // G) * q).to(torch.uint8)  # a[g*q + p]
    dd = d.to(torch.float16).view(torch.uint8).view(R, nSB, 2)
    blk = torch.cat([qs, a, dd], -1)
    return blk.reshape(R, -1).numpy()


def dequant_blocks(blk, q, G, C):
    """numpy reference dequant of our own packing (for self-check)."""
    R = blk.shape[0]
    nSB = C // 256
    bs = q * 32 + q * (256 // G) + 2
    b = torch.from_numpy(blk.reshape(R, nSB, bs).copy())
    qs = b[..., :q * 32].view(R, nSB, q, 32)
    a = b[..., q * 32:q * 32 + q * (256 // G)].float().view(R, nSB, 256 // G, q)
    d = b[..., -2:].contiguous().view(torch.float16).float().view(R, nSB)
    bits = (qs[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1  # [R,nSB,q,32,8]
    s = bits.view(R, nSB, q, 256).float() * 2 - 1
    alpha = (a * d[..., None, None]).repeat_interleave(G, 2)  # [R,nSB,256,q]
    return (s.permute(0, 1, 3, 2) * alpha).sum(-1).view(R, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)

    base = gguf.GGUFReader(a.base)
    ref = gguf.GGUFReader(a.ref)
    ref_t = {t.name: t for t in ref.tensors}
    repl = {}
    for f in sorted(run.glob("model.language_model.layers.*.pt")):
        st = torch.load(f)
        name = f.stem
        li = int(name.split(".")[3])
        sub = ".".join(name.split(".")[4:6])
        gname = f"blk.{li}.{HF2GGUF[sub]}.weight"
        idx = st["idx"]
        R, C = idx.shape
        rp, cp = row_perm(sub, R), col_perm(sub, C)
        if not torch.equal(cp, torch.arange(C)):
            assert st.get("col_perm_applied"), f"{name}: quantizer must work in GGUF column order"
        if st.get("layout") == "block":
            q, bc, br = st["q"], st["bc"], st["br"]
            # the V-head re-tiling moves whole 128-row heads, so row groups of br | 128 rows stay intact
            assert R % br == 0 and (torch.equal(rp, torch.arange(R)) or 128 % br == 0), (name, br)
            ng = R // br
            Tp = st["T"].view(ng, -1, q)[rp.view(ng, br)[:, 0] // br]  # permute row groups along with rows
            blk = colwise.pack_block(idx[rp], Tp, st["d"][rp], q, bc, br)
            deq = colwise.dequant_block_bytes(blk, q, bc, br, C)
            ref_q = colwise.dequant_block(idx.long(), st["T"].float(), st["d"].float(), q, bc, br)[rp]
            assert (ref_q - deq).abs().max() <= 1e-3 * ref_q.abs().max() + 1e-6, (gname, (ref_q - deq).abs().max())
            ty = getattr(T, f"SAB{q}_{bc}X{br}", None) or getattr(T, f"SAB{q}_{bc}x{br}")
            rt = ref_t[gname]
            refw = torch.from_numpy(np.ascontiguousarray(rt.data).view(np.int16)).view(torch.bfloat16).float().view(R, C)
            rel = ((deq - refw).norm() / refw.norm()).item()
            assert rel < 0.6, (gname, rel)
            repl[gname] = (blk, ty, rel)
            print(f"{gname:28s} q={q} tile={bc}x{br} {R}x{C} weight-rel-diff {rel:.4f}", flush=True)
            continue
        q, G = st["q"], st["G"]
        assert (q, G) in SA_TYPE and st["fmt"] == "u8" and st["mu"] is None, (name, q, G)
        code, d = st["alpha_store"]["code"], st["alpha_store"]["d"]
        # verify mapping against BF16 GGUF
        rt = ref_t[gname]
        assert rt.tensor_type == gguf.GGMLQuantizationType.BF16, rt.tensor_type
        refw = torch.from_numpy(np.ascontiguousarray(rt.data).view(np.int16)).view(torch.bfloat16).float()
        refw = refw.view(R, C)
        blk = pack_blocks(idx[rp], code[rp], d[rp], q, G)
        deq = dequant_blocks(blk, q, G, C)
        # packing must reproduce the quantizer's weights exactly
        Qq = gptq_bcq.dequant(st["alpha"].float(), None, idx.long(), q, G)[rp]
        assert (Qq - deq).abs().max() <= 1e-3 * Qq.abs().max(), (gname, (Qq - deq).abs().max())
        # relative distance to the reference weight in GGUF layout (should match the quantizer's error)
        rel = ((deq - refw).norm() / refw.norm()).item()
        assert rel < 0.6, (gname, rel)
        repl[gname] = (blk, SA_TYPE[(q, G)], rel)
        print(f"{gname:28s} q={q} G={G} {R}x{C} weight-rel-diff {rel:.4f}", flush=True)

    # tuned RMSNorm weights (finetune --train-norms); HF stores w with scale (1 + w), GGUF stores 1 + w
    norm_repl = {}
    if (run / "norms.pt").exists():
        for n, v in torch.load(run / "norms.pt").items():
            if n == "norm.weight":
                gname = "output_norm.weight"
            else:
                li = int(n.split(".")[1])
                gname = f"blk.{li}." + {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm"}[n.split(".")[2]] + ".weight"
            new = (1 + v.float()).numpy().astype(np.float32)
            old = np.array(ref_t[gname].data, dtype=np.float32)
            assert np.abs(new - old).max() < 0.25 * np.abs(old).max() + 1e-3, (gname, np.abs(new - old).max())
            norm_repl[gname] = new.reshape(old.shape)
        print("replacing", len(norm_repl), "norm tensors")

    w = gguf.GGUFWriter(a.out, arch=base.fields["general.architecture"].contents(), endianess=base.endianess)
    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st_ = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st_)
    w.add_key_value("general.quantization_note", "ShiftAdd/BCQ decoder linears (SA2/SA3/SA4)", gguf.GGUFValueType.STRING)
    datas = []
    for t in base.tensors:
        if t.name in repl:
            blk, ty, _ = repl[t.name]
            data = blk
        elif t.name in norm_repl:
            assert t.tensor_type == gguf.GGMLQuantizationType.F32
            data, ty = norm_repl[t.name], t.tensor_type
        else:
            data, ty = t.data, t.tensor_type
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, ty)
        datas.append(data)
    missing = set(repl) - {t.name for t in base.tensors}
    assert not missing, missing
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for d in datas:
        w.write_tensor_data(d, tensor_endianess=base.endianess)
    w.close()
    print("wrote", a.out, Path(a.out).stat().st_size / 1e9, "GB")


if __name__ == "__main__":
    main()
