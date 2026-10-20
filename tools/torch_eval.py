"""PyTorch KLD/PPL of a quantized run (fake-quant, same windows as llama-perplexity) on wiki and mix text.

  python tools/torch_eval.py <run_dir> [wiki,mix]   -> <run_dir>/torch_eval.json
Used for the method comparison against fake-quant baselines (uniform GPTQ / RTN) that have no GGUF type.
"""
import json, sys
from pathlib import Path
import torch
from saq.finetune import swap_in
from saq.model import MODEL, eval_windows, kld_eval, load, reference_cache, reference_logprobs
run = Path(sys.argv[1]); texts = (sys.argv[2] if len(sys.argv) > 2 else "wiki,mix").split(",")
tok, model = load(MODEL)
evals = {t: (lambda w: (w, reference_logprobs(model, w, reference_cache(t, 40))))(eval_windows(tok, t, 512, 40)) for t in texts}
model.model.visual = None
swap_in(model, run)
res = {t: kld_eval(model, w, ref) for t, (w, ref) in evals.items()}
print(json.dumps(res)); (run / "torch_eval.json").write_text(json.dumps(res, indent=1))
