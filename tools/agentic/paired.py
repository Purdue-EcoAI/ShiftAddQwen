"""Per-item scores of the agentic runs and a paired bootstrap against a reference model.
  items(label) -> {metric: {item_id: score in [0,1]}}   (BFCL: per item, averaged over repeat runs; τ-bench: per task,
  averaged over trials; AIME: per problem, averaged over seeds)
  paired(a, b) -> (mean difference in points, lo, hi) of the 95% bootstrap CI over the items both have."""
import glob, json, os, random
from pathlib import Path

R = Path(os.environ.get("SAQ_WORK", "work")) / "results/agentic"
SPLITS = {"non_live": "bfcl_non_live", "live": "bfcl_live", "multi_turn": "bfcl_multi_turn"}


def _bfcl(d):
    out = {}
    for run in sorted(glob.glob(f"{d}/bfcl*")):
        for split, key in SPLITS.items():
            per = {}
            for rf in glob.glob(f"{run}/result/llama-server-FC/{split}/BFCL_v4_*_result.json"):
                cat = os.path.basename(rf)[len("BFCL_v4_"):-len("_result.json")]
                sf = f"{run}/score/llama-server-FC/{split}/BFCL_v4_{cat}_score.json"
                if not os.path.exists(sf):
                    per = None; break
                ids = [json.loads(l)["id"] for l in open(rf) if l.strip()]
                with open(sf) as h:
                    h.readline(); failed = {json.loads(l)["id"] for l in h if l.strip()}
                per.update({i: 0.0 if i in failed else 1.0 for i in ids})
            if per:
                out.setdefault(key, []).append(per)
    return {k: {i: sum(p[i] for p in runs) / len(runs) for i in runs[0] if all(i in p for p in runs)} for k, runs in out.items()}


def _tau(d):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent)); from tau_missing import infra_error
    out = {}
    for env in ("retail", "airline"):
        by = {}
        for f in sorted(glob.glob(f"{d}/tau_{env}/*.json"), key=os.path.getmtime):
            try:
                eps = json.load(open(f))
            except Exception:
                continue
            for x in eps:
                if not infra_error(x):
                    by.setdefault(int(x["task_id"]), []).append(float(x["reward"]))
        if by:
            out[f"tau_{env}"] = {t: sum(v[:2]) / len(v[:2]) for t, v in by.items()}
    return out


def _aime(d):
    by = {}
    for f in glob.glob(f"{d}/reasoning*.aime25.s*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l); by.setdefault(r["id"], []).append(float(r["score"]))
    return {"aime25": {i: sum(v) / len(v) for i, v in by.items()}} if by else {}


def items(label):
    d = R / label
    return {**_bfcl(d), **_tau(d), **_aime(d)}


def paired(a, b, n=2000, seed=0):
    ids = sorted(set(a) & set(b))
    if len(ids) < 10:
        return None
    diff = [100 * (a[i] - b[i]) for i in ids]
    rng = random.Random(seed); m = len(diff)
    boots = sorted(sum(diff[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(diff) / m, boots[int(0.025 * n)], boots[int(0.975 * n)], m


if __name__ == "__main__":
    import sys
    ref = sys.argv[1]; A = items(ref)
    for l in sys.argv[2:]:
        B = items(l)
        for k in A:
            if k in B:
                p = paired(B[k], A[k])
                if p:
                    print(f"{l:14s} {k:16s} diff={p[0]:+5.1f}  95% CI [{p[1]:+5.1f}, {p[2]:+5.1f}]  n={p[3]}  -> " + ("WIN" if p[1] > 0 else "LOSS" if p[2] < 0 else "tie"))
