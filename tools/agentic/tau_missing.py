"""Missing / errored tau-bench episodes of one run.  Prints one line per trial count: '<k> <task ids>' = these tasks
need k more valid episodes (an episode that failed on an infrastructure error, e.g. a litellm timeout, does not count;
a model-side exception such as malformed tool-call JSON is a genuine reward-0 episode).
  python tools/agentic/tau_missing.py <label> <retail|airline> [trials=2]
"""
import glob, json, os, sys
INFRA = ("litellm.", "Timeout", "Connection", "httpx", "APIError", "InternalServerError")


def infra_error(r):
    e = r.get("info", {}).get("error") if isinstance(r.get("info"), dict) else None
    return bool(e) and any(k in str(e) for k in INFRA)



if __name__ == "__main__":
    label, env = sys.argv[1], sys.argv[2]; trials = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    d = f"{os.environ.get('SAQ_WORK', 'work')}/results/agentic/{label}/tau_{env}"
    n_tasks = {"retail": 115, "airline": 50}[env]
    valid = {}
    for f in glob.glob(f"{d}/*.json"):
        for r in json.load(open(f)):
            if not infra_error(r):
                valid[int(r["task_id"])] = valid.get(int(r["task_id"]), 0) + 1
    need = {}
    for t in range(n_tasks):
        k = trials - valid.get(t, 0)
        if k > 0:
            need.setdefault(k, []).append(t)
    for k in sorted(need):
        print(k, " ".join(map(str, need[k])))
