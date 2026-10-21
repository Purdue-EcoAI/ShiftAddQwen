"""Build results/summary.{json,md} and docs/results.md from a collected summary (tools/collect.py output).

  python tools/make_results.py [collected_summary.json] [mem_breakdown.json]
Run labels are mapped to public names by tools/names.py; labels in HIDE are left out of the public tables.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from names import display, short, is_ours

R = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "work/results/summary.json"
mem_src = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "work/results/mem_breakdown.json"
ORDER = ['Q8_0', 'Q4_K_M', 'v2g4.8_ft', 'UD-Q3_K_XL', 'v2g4.4q6_ft', 'v2g4.4_ft', 'Q3_K_M', 'v2g4.0_ft', 'UD-Q2_K_XL', 'v2g3.6_ft', 'UD-IQ3_XXS',
         'v2g3.2_ft', 'UD-IQ2_M', 'v2g3.0_ft', 'v2g2.6_ft']
HIDE = {'v2g4.4_ft', 'v2g4.8_ft'}  # v2g4.4_ft superseded by the re-planned v2g4.4q6_ft; v2g4.8_ft: 5-plane kernels are compute-bound (slower decode)
rows = {r['label']: r for r in json.load(open(src))}
ci_path = R / "results/task_ci.json"  # bootstrap 95% CI of the task mean (tools/task_ci.py)
ci = json.load(open(ci_path)) if ci_path.exists() else {}
out = []
for l in ORDER:
    if l in HIDE or l not in rows:
        continue
    r = rows[l]; c = r['suite_cats']
    pub = [c.get(k) for k in ('math', 'knowledge', 'code', 'multiling')]
    out.append(dict(model=display(l), short=short(l), id=l, ours=is_ours(l), file_gb=r['file_gb'], mix_kld=r['mix_kld'],
                    wiki_kld=r['wiki_kld'], wiki_ppl=r['wiki_ppl'],
                    tasks_public4=round(sum(pub) / 4, 1) if None not in pub else None,
                    tasks_ci=[ci[l]["lo"], ci[l]["hi"]] if l in ci else None,
                    gsm8k=c.get('math'), mmlu_pro=c.get('knowledge'), humaneval=c.get('code'), mgsm=c.get('multiling'),
                    decode_tok_s=r.get('tg128'), prefill_tok_s=r.get('pp512'), mtp_decode_tok_s=r.get('mtp_tok_s'),
                    mtp_accept=r.get('mtp_accept'), gpu_resident_gib=(r['gpu_mib'] / 1024 if r.get('gpu_mib') else None),
                    ablation=False))
mem = json.load(open(mem_src)) if mem_src.exists() else {}
json.dump({"baseline_repo": "unsloth/Qwen3.5-9B-MTP-GGUF", "rows": out,
           "bytes_gb": {display(k): v for k, v in mem.items() if k not in HIDE}}, open(R / "results/summary.json", "w"), indent=1)
f = lambda v, n: "—" if v is None else f"{v:.{n}f}"
L = ["| model | GGUF (GB) | KLD mixed ↓ | KLD wiki ↓ | PPL wiki ↓ | tasks ↑ | GSM8K | MMLU-Pro | HumanEval | MGSM | decode tok/s ↑ | prefill tok/s ↑ | decode+MTP tok/s ↑ |", "|" + "---|" * 13]
for r in out:
    name = f"**{r['model']}**" if r['ours'] else r['model']
    L.append(f"| {name} | {f(r['file_gb'],2)} | {f(r['mix_kld'],4)} | {f(r['wiki_kld'],4)} | {f(r['wiki_ppl'],3)} | {f(r['tasks_public4'],1)} | {f(r['gsm8k'],1)} | {f(r['mmlu_pro'],1)} | {f(r['humaneval'],1)} | {f(r['mgsm'],1)} | {f(r['decode_tok_s'],1)} | {f(r['prefill_tok_s'],0)} | {f(r['mtp_decode_tok_s'],1)} |")
table = "\n".join(L) + "\n"
(R / "results/summary.md").write_text(table)
doc = (R / "docs/results.md").read_text()
a = doc.index("| model |"); b = doc.index("*Table 1.")
(R / "docs/results.md").write_text(doc[:a] + table + "\n" + doc[b:])
print(table)
