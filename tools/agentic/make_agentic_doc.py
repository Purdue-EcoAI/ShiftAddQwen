"""Write the agentic / reasoning comparison (results/agentic.json from collect_agentic.py) into docs/results.md,
between the '<!-- agentic:start -->' / '<!-- agentic:end -->' markers (appended if absent), and results/agentic.md."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from names import display

R = Path(__file__).resolve().parents[2]
rows = json.load(open(R / "results/agentic.json"))
T1 = {r["model"]: r for r in json.load(open(R / "results/summary.json"))["rows"]}  # Table 1: size, KLD, A100 throughput
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent)); from paired import items, paired as _paired
from functools import lru_cache
REF = "Q4_K_M"
MARGIN = 2.0  # points: |difference| above this is a win / loss, otherwise a tie
ITEMS = {r["label"]: items(r["label"]) for r in rows}
_P = {}
def paired(a, b):  # memoised per (label, metric) pair
    k = (id(a), id(b))
    if k not in _P:
        _P[k] = _paired(a, b)
    return _P[k]


MET = {"non_live": "bfcl_non_live", "live": "bfcl_live", "multi_turn": "bfcl_multi_turn", "retail": "tau_retail", "airline": "tau_airline"}


def mark(label, metric):
    """paired bootstrap vs the reference on the same items: ▲ better / ▼ worse (95% CI excludes 0) / ≈ tie, with the
    per-item difference in points"""
    if label == REF:
        return "ref"
    a, b = ITEMS.get(label, {}).get(MET[metric]), ITEMS.get(REF, {}).get(MET[metric])
    if not a or not b:
        return ""
    pr = paired(a, b)
    if not pr:
        return ""
    d, lo, hi, n = pr
    return f"{'▲' if d > MARGIN else '▼' if d < -MARGIN else '≈'}{d:+.1f}"

ORDER = ["Q4_K_M", "v2g4.4q6_ft", "Q3_K_M", "v2g4.0_ft"]  # by GGUF size
rows = sorted(rows, key=lambda r: ORDER.index(r["label"]) if r["label"] in ORDER else 99)
f = lambda v, n=1: "—" if v is None else f"{v:.{n}f}"
name = lambda r: f"**{r['model']}**" if r["ours"] else r["model"]

L = ["| model | GGUF (GB) | KLD mixed ↓ | prefill tok/s ↑ | decode tok/s ↑ | BFCL non-live ↑ | BFCL live ↑ | BFCL multi-turn ↑ | τ-bench retail pass^1 ↑ | τ-bench airline pass^1 ↑ |", "|" + "---|" * 10]
for r in rows:
    t1 = T1.get(r["model"], {})
    b = lambda g: r["bfcl"].get(g, {}).get("overall"); t = lambda e: r["tau"].get(e, {})
    tau_cell = lambda e: "—" if not t(e) else f"{t(e)['pass1']:.1f} ({t(e)['episodes']} ep.)"
    have = {"non_live": b("non_live"), "live": b("live"), "multi_turn": b("multi_turn"), "retail": t("retail"), "airline": t("airline")}
    mk = lambda m: (f" {mark(r['label'], m)}" if have.get(m) and mark(r["label"], m) not in ("", "ref") else "")  # no mark on a pending cell
    bcell = lambda g: (f(b(g)) if len(r["bfcl"].get(g, {}).get("runs", [])) < 2 else f"{b(g):.1f} ({' / '.join(f'{x:.1f}' for x in r['bfcl'][g]['runs'])})") + mk(g)
    ref1 = T1.get(display(REF), {})
    rel = lambda k, n: (f" ({100 * (t1[k] / ref1[k] - 1):+.0f}%)" if r["label"] != REF and t1.get(k) and ref1.get(k) else "")
    L.append(f"| {name(r)} | {f(t1.get('file_gb'), 2)}{rel('file_gb', 2)} | {f(t1.get('mix_kld'), 4)} | {f(t1.get('prefill_tok_s'), 0)}{rel('prefill_tok_s', 0)} | {f(t1.get('decode_tok_s'), 1)}{rel('decode_tok_s', 1)} | {bcell('non_live')} | {bcell('live')} | {bcell('multi_turn')} | {tau_cell('retail')}{mk('retail')} | {tau_cell('airline')}{mk('airline')} |")
main = "\n".join(L)

COLS = [("non_live", "Python Simple AST", "simple (py)"), ("non_live", "Java Simple AST", "simple (java)"), ("non_live", "JavaScript Simple AST", "simple (js)"),
        ("non_live", "Multiple AST", "multiple"), ("non_live", "Parallel AST", "parallel"), ("non_live", "Parallel Multiple AST", "par. multiple"),
        ("non_live", "Irrelevance Detection", "irrelevance"), ("live", "Python Simple AST", "live simple"), ("live", "Python Multiple AST", "live multiple"),
        ("live", "Python Parallel AST", "live parallel"), ("live", "Python Parallel Multiple AST", "live par. mult."), ("live", "Irrelevance Detection", "live irrelevance"),
        ("live", "Relevance Detection", "live relevance"), ("multi_turn", "Base", "MT base"), ("multi_turn", "Miss Func", "MT miss func"),
        ("multi_turn", "Miss Param", "MT miss param"), ("multi_turn", "Long Context", "MT long ctx")]
D = ["| model | " + " | ".join(c[2] for c in COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
for r in rows:
    D.append(f"| {name(r)} | " + " | ".join(f(r["bfcl"].get(s, {}).get("cols", {}).get(c)) for s, c, _ in COLS) + " |")
detail = "\n".join(D)

T = ["| model | retail pass^1 | retail pass^2 | retail episodes | airline pass^1 | airline pass^2 | airline episodes |", "|" + "---|" * 7]
for r in rows:
    t = lambda e, k: r["tau"].get(e, {}).get(k)
    T.append(f"| {name(r)} | {f(t('retail','pass1'))} | {f(t('retail','pass_k'))} | {t('retail','episodes') or '—'} | {f(t('airline','pass1'))} | {f(t('airline','pass_k'))} | {t('airline','episodes') or '—'} |")
tau_tbl = "\n".join(T)


body = f"""<!-- agentic:start -->
## Agentic benchmarks: tool calling

{main}

*Table 2. Tool calling of the two drop-in candidates against the reference 4-bit deployment (and Q3_K_M, the standard 3-bit recipe of similar size), all served
by the same patched `llama-server` (`--jinja`, thinking on, temperature 0.6 / top-p 0.95). GGUF size, KLD to BF16 on the
mixed-domain text and `llama-bench` prefill (pp512) / decode (tg128) throughput on the A100 are repeated from Table 1, with
the change relative to Q4_K_M in parentheses. BFCL v4 (Berkeley Function
Calling Leaderboard): accuracy (%) of the non-live (1,390 items), live (2,251) and multi-turn (800) splits, scored by
`bfcl evaluate`; multi-turn is the mean of two generation runs (both values in parentheses). τ-bench: pass^1 (%) = mean task success over 2 trials of the retail (115 tasks) and airline (50 tasks)
environments; the user simulator is the same model with thinking off. ▲ / ▼ / ≈ after a score: better / worse than / within 2 points of
Q4_K_M, followed by the per-item difference in points (paired over the same BFCL items and τ-bench tasks;
it can differ from the difference of the BFCL overall numbers, which are BFCL's category-weighted means). Raw numbers and paired bootstrap 95% intervals: [`results/agentic.json`](../results/agentic.json).*

**BFCL per category** (accuracy, %):

{detail}

**τ-bench** (pass^k = fraction of tasks solved in all k trials):

{tau_tbl}

**Protocol.** `tools/agentic/run_agentic.sh` starts `llama-server` (8 parallel slots for BFCL, 16 for τ-bench; 32K context each) and runs BFCL (`bfcl generate` through the OpenAI chat-completions API with
`tools`, so the GGUF's own Qwen3.5 chat template and llama.cpp's XML tool-call parser are exercised; model entry
`llama-server-FC` from `tools/agentic/bfcl_local_model.py`) and τ-bench (`tool-calling` agent through litellm; user simulator patched by `tools/agentic/tau_user_nothink.py`). Hardware:
one NVIDIA V100-SXM2-32GB per (model, split). Sampling noise: BFCL splits are one generation per item at temperature 0.6,
so differences of about 1 point (non-live/live) to 2 points (multi-turn) are within run-to-run noise; τ-bench pass^1 over
230 / 100 episodes has a standard error of roughly 3 / 5 points.
<!-- agentic:end -->"""
for r in rows:
    r["vs_" + REF] = {m: dict(zip(("diff", "lo95", "hi95", "n"), paired(ITEMS[r["label"]][k], ITEMS[REF][k])))
                      for m, k in MET.items() if r["label"] != REF and ITEMS.get(r["label"], {}).get(k) and ITEMS.get(REF, {}).get(k) and paired(ITEMS[r["label"]][k], ITEMS[REF][k])}
json.dump(rows, open(R / "results/agentic.json", "w"), indent=1)
(R / "results/agentic.md").write_text(body + "\n")
doc = (R / "docs/results.md").read_text()
if "<!-- agentic:start -->" in doc:
    a, b = doc.index("<!-- agentic:start -->"), doc.index("<!-- agentic:end -->") + len("<!-- agentic:end -->")
    doc = doc[:a] + body + doc[b:]
else:
    doc = doc.rstrip("\n") + "\n\n" + body + "\n"
(R / "docs/results.md").write_text(doc)
print(main)
