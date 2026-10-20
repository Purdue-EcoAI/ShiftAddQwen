"""Score a suite against an OpenAI-compatible server (llama-server), greedy, thinking disabled.

  python tools/eval/run_eval.py --url http://127.0.0.1:8080 --suite suite.json --out res.json --label X

HumanEval completions are executed locally (isolated interpreter, temp dir, timeout) — run this
only in an environment where executing model-written code is acceptable.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

NUM = r"-?\d[\d,]*(?:\.\d+)?"


def visible(text):
    return text.split("</think>")[-1].strip()


def to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def score_number(item, out):
    m = re.findall(r"[Aa]nswer\s*[:：]\s*\**\s*\$?\s*(" + NUM + ")", out) or re.findall(r"\\boxed\{\s*(" + NUM + ")", out) \
        or re.findall(NUM, out)
    pred = to_float(m[-1]) if m else None
    return float(pred is not None and abs(pred - float(item["answer"])) < 1e-6)


def score_letter(item, out):
    valid = "ABCDEFGHIJ"[: item["n_options"]]
    m = re.search(r"[Aa]nswer\s*(?:is|:)?\s*\(?\**([A-J])\b", out) or re.match(r"^\W*([A-J])\b", out) \
        or re.search(r"\b([A-J])\b", out)
    return float(bool(m) and m.group(1) in valid and m.group(1) == item["answer"])


def score_code(item, out):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", out, re.S) or [out.split("```python")[-1]]
    code = next((b for b in blocks if f"def {item['entry_point']}" in b), blocks[0])
    prog = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.py"
        p.write_text(prog)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True, timeout=15)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0


SCORE = {"math": score_number, "multiling": score_number, "knowledge": score_letter, "code": score_code}


def ask(url, item):
    body = {"model": "default", "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": item["max_tokens"], "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=1800).json()
    return visible(r["choices"][0]["message"].get("content") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    items = json.loads(Path(a.suite).read_text())
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        outs = list(ex.map(lambda it: ask(a.url, it), items))
    res = [{"id": it["id"], "cat": it["cat"], "score": SCORE[it["cat"]](it, o), "output": o} for it, o in zip(items, outs)]
    cats = sorted({r["cat"] for r in res})
    summary = {c: round(100 * sum(r["score"] for r in res if r["cat"] == c) / sum(r["cat"] == c for r in res), 1) for c in cats}
    summary["macro_avg"] = round(sum(summary[c] for c in cats) / len(cats), 1)
    json.dump({"label": a.label, "summary": summary, "seconds": time.time() - t0, "results": res},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"label": a.label, "summary": summary}))


if __name__ == "__main__":
    main()
