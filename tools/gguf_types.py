import os, sys, collections, re
sys.path.insert(0, os.path.join(os.environ.get("LLAMA_CPP", "work/llama.cpp"), "gguf-py"))
import gguf
for f in sys.argv[1:]:
    r = gguf.GGUFReader(f)
    agg = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    for t in r.tensors:
        key = re.sub(r"blk\.\d+\.", "blk.N.", t.name)
        if t.name.startswith("blk.32."): key = "MTP:" + key
        agg[key][t.tensor_type.name] += 1
        tot[t.tensor_type.name] += int(t.n_bytes)
    print("==", f.split("/")[-1])
    for k, v in sorted(agg.items()): print(f"  {k:40s} {dict(v)}")
    print("  bytes by type:", {k: round(v/1e9, 3) for k, v in tot.items()})
