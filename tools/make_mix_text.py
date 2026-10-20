"""Held-out mixed-domain eval text for llama-perplexity (v2 mix, seed 3: web, chat, code, multilingual, math)."""
from transformers import AutoTokenizer
from saq import data
from saq.model import MODEL, WORK
tok = AutoTokenizer.from_pretrained(MODEL)
x = data.calibration(tok, 40, 2048, seed=3, version=2)
txt = "\n\n".join(tok.decode(s) for s in x)
(WORK / "data/mix.test.raw").write_text(txt)
print(len(txt))
