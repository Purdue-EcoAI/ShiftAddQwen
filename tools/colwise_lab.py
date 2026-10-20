"""Row-wise (this repo's format) vs column-wise (ShiftAddLLM acc-style) BCQ scales at equal bpw."""
import json, sys, torch
from saq import gptq_bcq
from saq.model import WORK
torch.backends.cuda.matmul.allow_tf32 = False
for f in sys.argv[1].split(","):
    d = torch.load(WORK / "data/lab" / f); W, H = d["W"].cuda().float(), d["H"].cuda().float()
    for q in (2, 3, 4):
        row = gptq_bcq.quantize(W, H, q, G=64, act_order=True)["err"]
        row_norefit = gptq_bcq.quantize(W, H, q, G=64, act_order=True, refit=False)["err"]
        col = gptq_bcq.quantize_colwise(W, H, q, GR=64)["err"]
        print(json.dumps(dict(file=f, q=q, row_refit=round(row, 6), row_norefit=round(row_norefit, 6), col=round(col, 6))), flush=True)
