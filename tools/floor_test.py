"""Is there a measurement floor in saq.sens_kld?  Perturb one linear by tiny / 8-bit / 4-bit error."""
import torch, torch.nn.functional as F
from saq import data, gptq_bcq
from saq.model import MODEL, WORK, load, quant_linears, text_layers, text_logits
from saq.sens_kld import kld_windows
tok, model = load(MODEL); model.model.visual = None
held = data.calibration(tok, 64, 2048, seed=2); win = held[:, :1024].reshape(-1, 512)[:24]
with torch.no_grad():
    ref16 = [F.log_softmax(text_logits(model, w[None].cuda())[0, 255:-1].float(), -1).half().cpu() for w in win]
    ref32 = [F.log_softmax(text_logits(model, w[None].cuda())[0, 255:-1].float(), -1).cpu() for w in win]
print("base fp16-ref", kld_windows(model, win, ref16), "base fp32-ref", kld_windows(model, win, ref32))
lin = quant_linears(text_layers(model)[24])["mlp.down_proj"]; W0 = lin.weight.data.clone()
for rel in (1e-3, 1e-2, 3e-2, 1e-1):
    lin.weight.data.copy_((W0.float() * (1 + rel * torch.randn_like(W0.float()))).to(W0.dtype))
    print("noise", rel, kld_windows(model, win, ref16) , kld_windows(model, win, ref32))
lin.weight.data.copy_(W0)
