"""Reasoning / instruction-following evaluation against an OpenAI-compatible server (llama-server).

  python tools/agentic/reasoning_eval.py --url http://127.0.0.1:8080 --out res.json --label X \
      [--tasks aime25,math500,gpqa,ifeval] [--seeds 4] [--think] [--max-tokens 32768]

Thinking mode on (Qwen3.5 default); sampling at the model card's recommended temperature 0.6 / top-p 0.95
with several seeds, so differences are averaged over sampling noise. Scoring: AIME/MATH-500 numeric or
boxed-expression match; GPQA letter match (options shuffled per seed); IFEval strict-prompt accuracy via
the official checker (vendored in ifeval_lib; needs nltk punkt_tab, langdetect, immutabledict).
"""
import argparse, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def last_boxed(t):
    m = BOX.findall(t)
    return m[-1].strip() if m else None


def norm_math(s):
    if s is None:
        return None
    s = s.strip().rstrip(".").replace("$", "").replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    return s


def ask(url, msgs, max_tokens, temp, seed, think):
    body = {"model": "default", "messages": msgs, "max_tokens": max_tokens, "temperature": temp, "top_p": 0.95,
            "seed": seed, "chat_template_kwargs": {"enable_thinking": think}}
    for attempt in range(3):
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=7200).json()
        except Exception as e:  # connection hiccup: retry
            r = {"error": str(e)}
        if "choices" in r:
            m = r["choices"][0]["message"]
            return (m.get("content") or ""), (m.get("reasoning_content") or ""), r.get("usage", {})
        err = r.get("error", r)
        if attempt == 2 or "context" in str(err).lower():
            print("request failed:", str(err)[:200], flush=True)
            return "", "", {"error": str(err)[:200]}
        time.sleep(5)


def items_for(task, seed):
    rng = random.Random(seed)
    if task == "aime25":
        return [dict(id=f"aime25-{r['id']}", prompt=r["problem"] + "\n\nPut the final answer in \\boxed{}.", answer=str(r["answer"]).strip(), kind="num")
                for r in load_dataset("math-ai/aime25", split="test")]
    if task == "math500":
        return [dict(id=f"math500-{r['unique_id']}", prompt=r["problem"] + "\n\nPut the final answer in \\boxed{}.", answer=r["answer"], kind="expr")
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    if task == "gpqa":
        out = []
        for i, r in enumerate(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4)); rng.shuffle(order)
            letters = "ABCD"; gold = letters[order.index(0)]
            txt = "\n".join(f"{letters[k]}) {opts[j].strip()}" for k, j in enumerate(order))
            out.append(dict(id=f"gpqa-{i}", prompt=f"{r['Question'].strip()}\n\n{txt}\n\nAnswer with the letter of the correct option in \\boxed{{}}.", answer=gold, kind="letter"))
        return out
    if task == "ifeval":
        return [dict(id=f"ifeval-{r['key']}", prompt=r["prompt"], instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"], kind="ifeval")
                for r in load_dataset("google/IFEval", split="train")]
    raise ValueError(task)


def score(it, content):
    if it["kind"] == "ifeval":
        from ifeval_lib import evaluate_instruction_following  # vendored official checker (tools/agentic/ifeval_lib)
        try:
            return float(evaluate_instruction_following(it["prompt"], content, it["instruction_id_list"], it["kwargs"], strict=True))
        except Exception as e:  # checker crash on one response counts as not followed
            print("ifeval checker error:", it["id"], str(e)[:100], flush=True)
            return 0.0
    pred = last_boxed(content)
    if it["kind"] == "letter":
        m = re.search(r"[A-D]", pred or "") or re.search(r"(?:answer|Answer)\s*(?:is|:)?\s*\(?([A-D])\b", content)
        return float(bool(m) and m.group(0)[-1] == it["answer"])
    if it["kind"] == "num":
        try:
            return float(abs(float(norm_math(pred)) - float(it["answer"])) < 1e-6)
        except (TypeError, ValueError):
            return 0.0
    return float(norm_math(pred) == norm_math(it["answer"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--tasks", default="aime25,math500,gpqa,ifeval"); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--think", action="store_true"); ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    res = {"label": a.label, "think": a.think, "seeds": a.seeds, "tasks": {}}
    t0 = time.time()
    for task in a.tasks.split(","):
        seeds = range(a.seeds) if task != "ifeval" else [0]
        per_seed, toks = [], []
        for s in seeds:
            items = items_for(task, s)
            if a.limit:
                items = items[: a.limit]
            with ThreadPoolExecutor(a.concurrency) as ex:
                outs = list(ex.map(lambda it: ask(a.url, [{"role": "user", "content": it["prompt"]}], a.max_tokens, a.temperature, s, a.think), items))
            sc = [score(it, c) for it, (c, _, _) in zip(items, outs)]
            sc = [x for x in sc if x is not None]
            per_seed.append(100 * sum(sc) / max(1, len(sc)))
            toks.append(sum(u.get("completion_tokens", 0) for _, _, u in outs) / max(1, len(outs)))
            Path(a.out).with_suffix(f".{task}.s{s}.jsonl").write_text("\n".join(json.dumps({"id": it["id"], "score": x, "content": c[-2000:], "usage": u}) for it, (c, _, u), x in zip(items, outs, [score(it, c) for it, (c, _, _) in zip(items, outs)])))
        res["tasks"][task] = {"mean": round(sum(per_seed) / len(per_seed), 2), "per_seed": [round(x, 2) for x in per_seed],
                              "avg_completion_tokens": round(sum(toks) / len(toks))}
        print(task, res["tasks"][task], flush=True)
    res["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
