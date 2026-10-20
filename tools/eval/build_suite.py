"""Build the task suite (public datasets only) from the fixed item list in items.json.

  python tools/eval/build_suite.py out/suite.json

Categories: GSM8K (math, numeric exact match), MMLU-Pro (knowledge, letter match),
HumanEval (code, unit tests), MGSM (multilingual math, numeric exact match).
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

ITEMS = json.loads((Path(__file__).parent / "items.json").read_text())
NUM_TAIL = "\n\nWork it out briefly, then give the result on a final line formatted as 'Answer: <number>'."
LETTERS = "ABCDEFGHIJ"


def main(out):
    items = []
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i in ITEMS["gsm8k"]:
        r = gsm[i]
        items.append({"id": f"gsm8k-{i}", "cat": "math", "prompt": r["question"] + NUM_TAIL,
                      "answer": r["answer"].split("####")[-1].strip().replace(",", ""), "max_tokens": 768})

    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_id = {qid: k for k, qid in enumerate(mp["question_id"])}
    for qid in ITEMS["mmlupro"]:
        r = mp[by_id[qid]]
        opts = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(r["options"]))
        items.append({"id": f"mmlupro-{qid}", "cat": "knowledge", "n_options": len(r["options"]),
                      "prompt": f"Multiple-choice question ({r['category']}).\n\n{r['question']}\n\n{opts}\n\n"
                                "Reply with the letter of the correct option only.",
                      "answer": r["answer"], "max_tokens": 16})

    he = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test")}
    for tid in ITEMS["humaneval"]:
        r = he[tid]
        items.append({"id": tid, "cat": "code", "entry_point": r["entry_point"], "test": r["test"],
                      "prompt": "Write the complete Python function below (signature, imports and body) "
                                f"in one ```python code block.\n\n```python\n{r['prompt']}```",
                      "max_tokens": 768})

    mg = {}
    for lang, i in ITEMS["mgsm"]:
        if lang not in mg:
            mg[lang] = load_dataset("juletxara/mgsm", lang, split="test")
        r = mg[lang][i]
        a = float(r["answer_number"])
        items.append({"id": f"mgsm-{lang}-{i}", "cat": "multiling", "prompt": r["question"] + NUM_TAIL,
                      "answer": str(int(a)) if a.is_integer() else str(a), "max_tokens": 768})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(len(items), "items ->", out)


if __name__ == "__main__":
    main(sys.argv[1])
