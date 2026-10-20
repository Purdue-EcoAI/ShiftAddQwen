"""Calibration and evaluation text.

Calibration mixes raw web/wiki text with chat-formatted general, code, multilingual and math
conversations (v2), so the Hessians and the distillation data see the token distribution an
assistant model actually serves. Evaluation uses wikitext-2 *test*, cut into the
same 512-token windows llama-perplexity uses.
"""
import random
from pathlib import Path

import torch
from datasets import load_dataset

from .model import WORK

ROOT = WORK / "data"


def _raw_wiki_train():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return "".join(ds["text"])


def _c4_docs(n):
    ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                      split="validation")
    return [ds[i]["text"] for i in range(n)]


def _chat_docs(tok, n, rng):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    out = []
    for i in rng.sample(range(len(ds)), n):
        out.append(tok.apply_chat_template(ds[i]["messages"], tokenize=False))
    return out


def _code_docs(tok, n, rng):
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        user = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        out.append(tok.apply_chat_template([{"role": "user", "content": user},
                                            {"role": "assistant", "content": r["output"]}], tokenize=False))
    return out


ML_LANGS = ["Simplified Chinese", "Traditional Chinese", "Japanese", "Korean", "Spanish", "French", "German",
            "Russian", "Thai", "Swahili", "Standard Arabic", "Portuguese", "Vietnamese", "Hindi", "Turkish",
            "Italian", "Indonesian", "Bengali", "Iranian Persian", "Ukrainian"]


def _multilingual_docs(tok, n, rng):
    """Aya human-written prompts/answers, balanced over widely used languages, as chat."""
    ds = load_dataset("CohereForAI/aya_dataset", split="train")
    by = {}
    for i, lang in enumerate(ds["language"]):
        if lang in ML_LANGS:
            by.setdefault(lang, []).append(i)
    out = []
    per = n // len(by) + 1
    for lang, idx in by.items():
        for i in rng.sample(idx, min(per, len(idx))):
            r = ds[i]
            out.append(tok.apply_chat_template([{"role": "user", "content": r["inputs"]},
                                                {"role": "assistant", "content": r["targets"]}], tokenize=False))
    rng.shuffle(out)
    return out


def _math_docs(tok, n, rng):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    out = []
    for i in rng.sample(range(len(ds)), n):
        r = ds[i]
        out.append(tok.apply_chat_template([{"role": "user", "content": r["query"]},
                                            {"role": "assistant", "content": r["response"]}], tokenize=False))
    return out


def _pack(texts, tok, nseq, seqlen, rng):
    """Concatenate shuffled documents (EOS separated) and cut nseq windows of seqlen."""
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id)
    ids = torch.tensor(ids)
    assert len(ids) > nseq * seqlen, (len(ids), nseq * seqlen)
    starts = rng.sample(range(0, len(ids) - seqlen), nseq)
    return torch.stack([ids[s:s + seqlen] for s in starts])


def calibration(tok, nseq=128, seqlen=2048, seed=0, mix=(0.375, 0.375, 0.25), version=1):
    if version == 2:
        return calibration_v2(tok, nseq, seqlen, seed)
    cache = ROOT / f"calib_{nseq}x{seqlen}_s{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    rng = random.Random(seed)
    n_raw, n_chat = round(nseq * mix[0]), round(nseq * mix[1])
    n_code = nseq - n_raw - n_chat
    wiki = _raw_wiki_train()
    c4 = _c4_docs(3000)
    raw_texts = [wiki] + c4
    rng.shuffle(raw_texts)
    chat = _chat_docs(tok, 1500, rng)
    code = _code_docs(tok, 4000, rng)
    x = torch.cat([_pack(raw_texts, tok, n_raw, seqlen, rng),
                   _pack(chat, tok, n_chat, seqlen, rng),
                   _pack(code, tok, n_code, seqlen, rng)])
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, cache)
    return x


def calibration_v2(tok, nseq, seqlen, seed):
    """v2 mix: raw web/wiki 25%, English chat 20%, code 15%, multilingual chat 25%, math chat 15%."""
    cache = ROOT / f"calib2_{nseq}x{seqlen}_s{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    rng = random.Random(seed)
    fr = [0.25, 0.20, 0.15, 0.25, 0.15]
    ns = [round(nseq * f) for f in fr]
    ns[0] += nseq - sum(ns)
    raw = [_raw_wiki_train()] + _c4_docs(3000)
    rng.shuffle(raw)
    parts = [_pack(raw, tok, ns[0], seqlen, rng),
             _pack(_chat_docs(tok, 1500, rng), tok, ns[1], seqlen, rng),
             _pack(_code_docs(tok, 4000, rng), tok, ns[2], seqlen, rng),
             _pack(_multilingual_docs(tok, 12000, rng), tok, ns[3], seqlen, rng),
             _pack(_math_docs(tok, 6000, rng), tok, ns[4], seqlen, rng)]
    x = torch.cat(parts)
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, cache)
    return x


def wiki_test_windows(tok, ctx=512, chunks=40):
    """Same chunking as llama-perplexity: consecutive ctx windows of the tokenized test set."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
    n = min(chunks, len(ids) // ctx)
    return ids[: n * ctx].view(n, ctx)
