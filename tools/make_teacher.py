"""Precompute the distillation data (calibration seed 1) and the bf16 teacher's top-64 log-probs."""
import sys
from saq import data
from saq.finetune import teacher_topk
from saq.model import MODEL, WORK, load
n = int(sys.argv[1]); v = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tok, model = load(MODEL)
seqs = data.calibration(tok, n, 2048, seed=1, version=v)
tag = "" if v == 1 else f"v{v}_"
teacher_topk(model, seqs, 64, WORK / f"data/teacher_{tag}top64_{n}x2048_s1.pt")
data.calibration(tok, 128, 2048, seed=0, version=v)  # GPTQ calibration set
print("done")
