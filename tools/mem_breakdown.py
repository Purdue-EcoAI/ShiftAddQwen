"""Bytes per tensor category in a GGUF (decoder linears / token_embd / output head / MTP block / rest)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", "work/llama.cpp"), "gguf-py"))
import gguf
out = {}
for kv in sys.argv[1:]:
    lab, f = kv.split("=")
    r = gguf.GGUFReader(f); c = {"linears": 0, "token_embd": 0, "output": 0, "mtp": 0, "other": 0}
    for t in r.tensors:
        n = t.name; b = int(t.n_bytes)
        if n.startswith("blk.32."): c["mtp"] += b
        elif n.startswith("token_embd"): c["token_embd"] += b
        elif n.startswith("output."): c["output"] += b
        elif n.startswith("blk.") and n.endswith(".weight") and any(k in n for k in ("attn_qkv", "attn_gate", "ssm_out", "attn_q.", "attn_k.", "attn_v.", "attn_output", "ffn_")): c["linears"] += b
        else: c["other"] += b
    out[lab] = {k: round(v / 1e9, 3) for k, v in c.items()}
    print(lab, out[lab])
json.dump(out, open(os.path.join(os.environ.get("SAQ_WORK", "work"), "results/mem_breakdown.json"), "w"), indent=1)
