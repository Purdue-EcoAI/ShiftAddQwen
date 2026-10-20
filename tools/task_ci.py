"""Bootstrap 95% CI of the task-suite mean per model -> results/task_ci.json (from tools/eval outputs)."""
import glob, json, os, random, sys
cats = ["math", "knowledge", "code", "multiling"]
out = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.suite.json"))):
    lab = os.path.basename(f).split(".suite")[0]
    d = json.load(open(f)); by = {c: [r["score"] for r in d["results"] if r["cat"] == c] for c in cats}
    rng = random.Random(0); ms = sorted(sum(sum(rng.choice(by[c]) for _ in by[c]) / len(by[c]) for c in cats) / 4 * 100 for _ in range(4000))
    out[lab] = {"mean": round(sum(sum(by[c]) / len(by[c]) for c in cats) / 4 * 100, 2), "lo": round(ms[100], 2), "hi": round(ms[3899], 2)}
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=0))
