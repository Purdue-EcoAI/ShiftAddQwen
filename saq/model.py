"""Qwen3.5 loading, linear-layer bookkeeping and fast PyTorch KLD / PPL evaluation."""
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

# working directory for models, calibration caches, runs and results (large; keep it out of git)
WORK = Path(os.environ.get("SAQ_WORK", Path(__file__).resolve().parents[1] / "work"))
MODEL = os.environ.get("SAQ_MODEL", str(WORK / "models/Qwen3.5-9B"))

# Linears kept in high precision (tiny + numerically sensitive gate projections of Gated DeltaNet)
KEEP = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")

# Modules that read the same input tensor share one Hessian (first one is the representative).
SHARED_INPUT = [
    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("mlp.gate_proj", "mlp.up_proj"),
]
# true-sequential stages inside a decoder layer
STAGES = [
    ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("linear_attn.out_proj", "self_attn.o_proj"),
    ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp.down_proj",),
]


def gguf_col_perm(sub, C):
    """llama.cpp's Qwen3.5 converter re-tiles V heads; for out_proj that permutes input columns.
    Returns GGUF-order -> HF column index map (or None)."""
    if sub != "linear_attn.out_proj":
        return None
    nk, per_k, hd = 16, 2, 128
    assert C == nk * per_k * hd
    return torch.arange(C).reshape(nk, per_k, hd).permute(1, 0, 2).reshape(C)


def load(path, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    model.eval()
    model.config.use_cache = False
    return tok, model


def text_layers(model):
    return model.model.language_model.layers


def quant_linears(layer):
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear) and not n.endswith(KEEP)}


class _Stop(Exception):
    pass


@torch.no_grad()
def run_until(model, layer_idx, batches, pre_hooks):
    """Forward each batch but abort right after decoder layer `layer_idx`."""
    layers = text_layers(model)

    def stop(*_):
        raise _Stop

    hs = [layers[layer_idx].register_forward_hook(stop)]
    hs += [m.register_forward_pre_hook(f) for m, f in pre_hooks]
    try:
        for b in batches:
            try:
                model.model.language_model(input_ids=b.cuda(), use_cache=False)
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()


def text_logits(model, ids):
    out = model.model.language_model(input_ids=ids, use_cache=False)
    return model.lm_head(out.last_hidden_state)


def eval_windows(tok, text="wiki", ctx=512, chunks=40):
    """Consecutive ctx-token windows of an evaluation text, as llama-perplexity cuts them.
    text = 'wiki' (wikitext-2 test) or 'mix' (held-out mixed-domain text, WORK/data/mix.test.raw)."""
    from . import data
    if text == "wiki":
        return data.wiki_test_windows(tok, ctx, chunks)
    raw = (WORK / "data" / f"{text}.test.raw").read_text()
    ids = torch.tensor(tok(raw, add_special_tokens=False)["input_ids"])
    n = min(chunks, len(ids) // ctx)
    return ids[: n * ctx].view(n, ctx)


def reference_cache(text, chunks):
    return WORK / "data" / (f"ref_lp_{chunks}.pt" if text == "wiki" else f"ref_lp_{text}_{chunks}.pt")


@torch.no_grad()
def reference_logprobs(model, windows, cache):
    """Store fp16 log-probs of the scored half of every window (llama-perplexity scores the 2nd half)."""
    if Path(cache).exists():
        return torch.load(cache, mmap=True)
    ctx = windows.shape[1]
    out = []
    for w in windows:
        lp = F.log_softmax(text_logits(model, w[None].cuda())[0, ctx // 2 - 1:-1].float(), -1)
        out.append(lp.half().cpu())
    out = torch.stack(out)
    torch.save(out, cache)
    return out


@torch.no_grad()
def kld_eval(model, windows, ref):
    """Mean KLD(ref || model), PPL and top-1 agreement over the scored halves."""
    ctx = windows.shape[1]
    kl, nll, agree, n = 0.0, 0.0, 0, 0
    kls = []
    for w, r in zip(windows, ref):
        lp = F.log_softmax(text_logits(model, w[None].cuda())[0, ctx // 2 - 1:-1].float(), -1)
        r = r.cuda().float()
        k = (r.exp() * (r - lp)).sum(-1)
        kls.append(k)
        tgt = w[ctx // 2:].cuda()
        nll += -lp.gather(-1, tgt[:, None]).sum().item()
        agree += (lp.argmax(-1) == r.argmax(-1)).sum().item()
        n += len(tgt)
    kls = torch.cat(kls)
    return {"kld": kls.mean().item(), "kld_p99": kls.quantile(0.99).item(),
            "ppl": math.exp(nll / n), "top1": agree / n, "tokens": n}
