"""Collect the agentic / reasoning runs (tools/agentic/run_agentic.sh) into results/agentic.json + a markdown table.
  python tools/agentic/collect_agentic.py [label ...]      (default: every label under $SAQ_WORK/results/agentic)
"""
import csv, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from names import display, is_ours
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tau_missing import infra_error

R = Path(os.environ.get("SAQ_WORK", "work")) / "results/agentic"
BFCL_CSV = {"non_live": ("data_non_live.csv", "Non-Live Overall Acc"), "live": ("data_live.csv", "Live Overall Acc"),
            "multi_turn": ("data_multi_turn.csv", "Multi Turn Overall Acc")}


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def bfcl(d):
    """BFCL's own score tables (bfcl evaluate), averaged over repeat runs (bfcl/, bfcl2/, ...): overall accuracy per
    split (+ the per-run values) and every column, as percentages"""
    pct = lambda v: None if v in (None, "N/A", "") else float(v.rstrip("%"))
    out = {}
    for split, (fn, col) in BFCL_CSV.items():
        runs = []
        for f in sorted(glob.glob(f"{d}/bfcl*/score/{fn}")):
            rows = list(csv.DictReader(open(f)))
            if rows and rows[0].get(col) not in (None, "0.00%", "N/A"):
                runs.append({k: pct(v) for k, v in rows[0].items() if k not in ("Rank", "Model")})
        if not runs:
            continue
        mean = lambda k: (sum(r[k] for r in runs if r.get(k) is not None) / len(runs)) if all(r.get(k) is not None for r in runs) else None
        out[split] = dict(overall=mean(col), runs=[r[col] for r in runs], cols={k: mean(k) for k in runs[0]})
    return out


def tau(d):
    out = {}
    for env in ("retail", "airline"):
        res = []
        for f in sorted(glob.glob(f"{d}/tau_{env}/*.json"), key=os.path.getmtime):
            r = load(f)
            if isinstance(r, list):  # infrastructure errors (litellm timeout etc.) are not scored; tau_missing.py reruns them
                res += [x for x in r if isinstance(x, dict) and "reward" in x and not infra_error(x)]
        if not res:
            continue
        by_task = {}
        for x in res:  # files are read in chronological order; keep the first K valid episodes per task
            by_task.setdefault(x["task_id"], []).append(x["reward"])
        K = 2
        by_task = {t: v[:K] for t, v in by_task.items()}  # trials per task (SEEDS in run_agentic.sh); pass^2 only over tasks that have both valid episodes
        pass1 = 100 * sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)
        full = [v for v in by_task.values() if len(v) >= K]
        pass_k = 100 * sum(all(r >= 1 for r in v[:K]) for v in full) / len(full) if full else None
        out[env] = dict(pass1=pass1, pass_k=pass_k, k=K, tasks=len(by_task), tasks_k=len(full), episodes=sum(len(v) for v in by_task.values()))
    return out


labels = sys.argv[1:] or sorted(p.name for p in R.iterdir() if p.is_dir())
rows = []
for l in labels:
    d = R / l
    tasks, seeds = {}, None  # one reasoning*.json per task set (run_agentic.sh REASONING_TASKS)
    for f in sorted(glob.glob(f"{d}/reasoning*.json")):
        rs = load(f) or {}
        tasks.update(rs.get("tasks", {})); seeds = rs.get("seeds", seeds)
    rows.append(dict(label=l, model=display(l), ours=is_ours(l), reasoning=tasks, seeds=seeds,
                     bfcl=bfcl(d), tau=tau(d)))
Path("results").mkdir(exist_ok=True)
json.dump(rows, open("results/agentic.json", "w"), indent=1)
f = lambda v, n=1: "—" if v is None else f"{v:.{n}f}"
T = ["| model | BFCL non-live | BFCL live | BFCL multi-turn | τ-retail pass^1 | τ-airline pass^1 | AIME25 | MATH-500 | IFEval |",
     "|" + "---|" * 9]
for r in rows:
    name = f"**{r['model']}**" if r["ours"] else r["model"]
    rt = lambda t: r["reasoning"].get(t, {}).get("mean")
    bo = lambda g: r["bfcl"].get(g, {}).get("overall")
    tp = lambda e: r["tau"].get(e, {}).get("pass1")
    T.append(f"| {name} | {f(bo('non_live'))} | {f(bo('live'))} | {f(bo('multi_turn'))} | {f(tp('retail'))} | {f(tp('airline'))} | "
             f"{f(rt('aime25'))} | {f(rt('math500'))} | {f(rt('ifeval'))} |")
print("\n".join(T))
