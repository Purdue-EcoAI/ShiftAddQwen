"""Gather every measured model into work/results/summary.json (+ markdown table on stdout)."""
import glob
import json
import os
import sys

R = os.path.join(os.environ.get("SAQ_WORK", "work"), "results")


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def bench(label):
    lb = load(f"{R}/bench/{label}.llama-bench.json")
    s = load(f"{R}/bench/{label}.server.json")
    m = load(f"{R}/bench/{label}.server.mtp.json")
    if not lb:
        return {}
    out = {"pp512": next(r["avg_ts"] for r in lb if r["n_prompt"] > 0),
           "tg128": next(r["avg_ts"] for r in lb if r["n_gen"] > 0)}
    if s:
        out["srv_tok_s"] = sum(r["predicted_per_second"] for r in s["requests"]) / len(s["requests"])
        out["gpu_mib"] = s.get("gpu_mem_mib")
    if m:
        out["mtp_tok_s"] = sum(r["predicted_per_second"] for r in m["requests"]) / len(m["requests"])
        out["mtp_accept"] = sum(r["draft_n_accepted"] for r in m["requests"]) / max(1, sum(r["draft_n"] for r in m["requests"]))
        out["gpu_mib_mtp"] = m.get("gpu_mem_mib")
    return out


labels = sys.argv[1:] or sorted({os.path.basename(f).split(".wiki.c40.json")[0] for f in glob.glob(f"{R}/kld/*.wiki.c40.json")})
rows = []
for l in labels:
    w = load(f"{R}/kld/{l}.wiki.c40.json") or {}
    mx = load(f"{R}/kld/{l}.mix.c40.json") or {}
    su = (load(f"{R}/suite/{l}.suite.json") or {}).get("summary", {})
    rows.append(dict(label=l, file_gb=w.get("file_gb"), wiki_kld=w.get("kld"), wiki_ppl=w.get("ppl"),
                     wiki_top1=w.get("top1"), mix_kld=mx.get("kld"), mix_top1=mx.get("top1"),
                     suite=su.get("macro_avg"), suite_cats=su, **bench(l)))
json.dump(rows, open(f"{R}/summary.json", "w"), indent=1)
f = lambda v, n=3: "—" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
print("| model | file GB | wiki KLD | wiki PPL | mix KLD | suite | math | know | code | multi | tg128 | pp512 | MTP tok/s | GPU GiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -(r["file_gb"] or 0)):
    c = r["suite_cats"]
    print(f"| {r['label']} | {f(r['file_gb'], 2)} | {f(r['wiki_kld'], 4)} | {f(r['wiki_ppl'], 3)} | {f(r['mix_kld'], 4)} | "
          f"{f(r['suite'], 1)} | {f(c.get('math'), 1)} | {f(c.get('knowledge'), 1)} | {f(c.get('code'), 1)} | {f(c.get('multiling'), 1)} | "
          f"{f(r.get('tg128'), 1)} | {f(r.get('pp512'), 0)} | {f(r.get('mtp_tok_s'), 1)} | "
          f"{f(r['gpu_mib'] / 1024 if r.get('gpu_mib') else None, 2)} |")
