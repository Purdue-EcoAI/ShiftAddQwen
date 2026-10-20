# ShiftAddQwen

**ShiftAddQwen** applies the [ShiftAddLLM](https://arxiv.org/abs/2406.05981) reparameterization to
Qwen3.5-9B (hybrid Gated-DeltaNet / full attention, with MTP head) and deploys it through llama.cpp.
Every decoder weight group is a sum of binary planes with per-group scales,
`w ≈ Σ_p α_p · b_p`, `b_p ∈ {−1, +1}`, so a mat-vec reduces to sign-select-and-add per plane plus one
scale per group. We contribute (i) an output-aware fit of this form for a hybrid-attention model,
(ii) KLD-measured mixed precision and end-to-end scale distillation, and (iii) new `ggml` quant types
with CUDA kernels, so the models run unmodified in llama.cpp (patched).

**Naming.** Our models are **ShiftAddQwen-3.5-9B-*x*bits**, *x* being the average bits per decoder
weight. Baselines are **UnslothAI/Qwen-3.5-9B-*recipe***: the public GGUF files of the same checkpoint
from [`unsloth/Qwen3.5-9B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF), i.e. the
standard llama.cpp recipes (`Q8_0`, `Q4_K_M`, `Q3_K_M`) and Unsloth's imatrix-calibrated dynamic
recipes (`UD-Q3_K_XL`, `UD-IQ3_XXS`, `UD-Q2_K_XL`, `UD-IQ2_M`). `Q4_K_M` (5.87 GB) is the reference
4-bit deployment.

## Results

![frontier](docs/frontier.png)

*Figure 1. ShiftAddQwen-3.5-9B-*x*bits (*x* = average bits per decoder weight) against the
UnslothAI/Qwen-3.5-9B-* GGUF quants. (a) KL divergence to the BF16 model on held-out mixed-domain text; (b) mean accuracy on
GSM8K, MMLU-Pro, HumanEval and MGSM; (c) decode throughput (A100, CUDA). Vector: [docs/frontier.pdf](docs/frontier.pdf).*

| model | GGUF (GB) | KLD mixed ↓ | KLD wiki ↓ | PPL wiki ↓ | tasks ↑ | GSM8K | MMLU-Pro | HumanEval | MGSM | decode tok/s ↑ | prefill tok/s ↑ | decode+MTP tok/s ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UnslothAI/Qwen-3.5-9B-Q8_0 | 9.79 | 0.0008 | 0.0013 | 7.074 | 77.9 | 94.0 | 51.7 | 84.8 | 81.2 | 107.0 | 4056 | 169.4 |
| UnslothAI/Qwen-3.5-9B-Q4_K_M | 5.87 | 0.0257 | 0.0289 | 7.109 | 79.2 | 94.4 | 52.6 | 87.2 | 82.5 | 125.2 | 4227 | 158.2 |
| UnslothAI/Qwen-3.5-9B-UD-Q3_K_XL | 5.24 | 0.0362 | 0.0447 | 7.274 | 78.9 | 94.0 | 54.9 | 82.9 | 83.8 | 115.8 | 3789 | 141.0 |
| **ShiftAddQwen-3.5-9B-4.4bits** | 5.11 | 0.0309 | 0.0365 | 7.253 | 77.1 | 92.4 | 50.6 | 84.1 | 81.2 | 134.9 | 5536 | 188.9 |
| UnslothAI/Qwen-3.5-9B-Q3_K_M | 4.83 | 0.0669 | 0.0787 | 7.206 | 76.2 | 92.4 | 50.3 | 84.8 | 77.5 | 111.0 | 3529 | 140.7 |
| **ShiftAddQwen-3.5-9B-4.0bits** | 4.76 | 0.0384 | 0.0458 | 7.259 | 77.4 | 92.4 | 49.7 | 84.8 | 82.5 | 134.6 | 5367 | 171.5 |
| UnslothAI/Qwen-3.5-9B-UD-Q2_K_XL | 4.44 | 0.1369 | 0.1564 | 7.940 | 72.8 | 92.4 | 47.4 | 77.4 | 73.8 | 126.1 | 3288 | 140.1 |
| **ShiftAddQwen-3.5-9B-3.6bits** | 4.41 | 0.0564 | 0.0655 | 7.325 | 75.3 | 94.0 | 48.6 | 82.3 | 76.2 | 137.4 | 5344 | 173.6 |
| UnslothAI/Qwen-3.5-9B-UD-IQ3_XXS | 4.30 | 0.0880 | 0.0979 | 7.502 | 74.8 | 92.4 | 46.9 | 81.1 | 78.8 | 133.6 | 4120 | 166.9 |
| **ShiftAddQwen-3.5-9B-3.2bits** | 4.07 | 0.0857 | 0.0974 | 7.539 | 74.8 | 91.2 | 50.0 | 80.5 | 77.5 | 142.7 | 5319 | 177.0 |
| UnslothAI/Qwen-3.5-9B-UD-IQ2_M | 3.97 | 0.1501 | 0.1649 | 7.921 | 70.7 | 90.0 | 42.3 | 78.0 | 72.5 | 135.4 | 3415 | 157.1 |
| **ShiftAddQwen-3.5-9B-3.0bits** | 3.90 | 0.1052 | 0.1230 | 7.678 | 73.2 | 88.8 | 49.7 | 80.5 | 73.8 | 144.8 | 5342 | 175.4 |
| **ShiftAddQwen-3.5-9B-2.6bits** | 3.55 | 0.1701 | 0.1973 | 8.204 | 68.0 | 84.8 | 42.3 | 71.3 | 73.8 | 151.0 | 5379 | 180.9 |

*Table 1. Full measurements; `results/summary.json` holds the raw numbers. "tasks" is the mean of the
four task columns.*

**Findings.**
1. *Memory.* ShiftAddQwen-3.5-9B-4.0bits is 19% smaller than UnslothAI/Qwen-3.5-9B-Q4_K_M (4.76 vs 5.87 GB) at near-equal
   quality (KLD 0.038 vs 0.026; tasks within noise). ShiftAddQwen-3.5-9B-3.6bits is 25% smaller and matches
   UnslothAI/Qwen-3.5-9B-Q3_K_M in task accuracy at lower KLD (0.056 vs 0.067).
2. *Fidelity frontier.* At every size below 5.2 GB, ShiftAddQwen has lower KLD than the Unsloth quant
   of equal or larger size (e.g. 0.086 at 4.07 GB vs UnslothAI/Qwen-3.5-9B-UD-IQ3_XXS 0.088 at 4.30 GB; 0.105 at 3.90 GB vs
   UnslothAI/Qwen-3.5-9B-UD-IQ2_M 0.150 at 3.97 GB).
3. *Speed.* The new kernels decode 7–15% faster than Q4_K_M and prefill 27% faster; MTP speculative
   decoding keeps its acceptance rate (0.60–0.68).

**Protocol.** KLD and PPL: `llama-perplexity` against the BF16 GGUF, 512-token windows, 40 chunks, on
wikitext-2 test ("wiki") and on held-out web / chat / code / multilingual / math text ("mixed",
`tools/make_mix_text.py`). Tasks: greedy decoding, thinking disabled, via `llama-server`, on GSM8K
(250), MMLU-Pro (350), HumanEval (164) and MGSM (80) items (`tools/eval`); one run per model, so
differences below ~1.5 points are noise (Q8_0 scores below Q4_K_M). Speed: NVIDIA A100-PCIE-40GB,
`llama-bench` pp512 / tg128, and `llama-server` at 65K context, with and without
`--spec-type draft-mtp --spec-draft-n-max 3`.

## Method

**Format.** `ggml` types `sa{2,3,4,5,6}[_128|_256]` (2.1–6.8 bits/weight): a super-block of 256 weights
along the input dimension stores `Q` sign planes (`Q·32` bytes), one u8 scale code per plane and
group of `G ∈ {64, 128, 256}` weights, and one fp16 scale `d`; `α = d·a`. The CUDA mat-vec turns
each nibble of sign bits into a byte mask and accumulates with `dp4a` against q8 activations, applying
the scale once per group and plane; prefill dequantizes to fp16 and uses cuBLAS. A generalized
block-wise family `sab<Q>_<bc>x<br>` (scales shared by `bc` input columns × `br` output rows, covering
ShiftAddLLM's column-wise and block-wise designs) is provided as well.

**Output-aware fit** (`saq/gptq_bcq.py`, `saq/colwise.py`). A GPTQ column sweep with act-order over
groups; at each group boundary the BCQ scales are fitted on the error-compensated weights (greedy
initialisation, alternating least squares with exhaustive code choice), and after the sweep all scales
are re-solved exactly for the Hessian-weighted output error with the codes frozen. Layers are
quantized sequentially on the outputs of the already-quantized prefix.

**Mixed precision** (`saq/sens_kld.py`, `saq/sensitivity.py`). The end-to-end KLD increase of
quantizing each linear alone at every (bits, group) setting is measured, the bf16 forward-noise floor
(≈3·10⁻⁴) subtracted, and a Lagrangian knapsack selects the setting per linear for a target average
bits/weight.

**Scale distillation** (`saq/finetune.py`, `saq/qlinear.py`). With the binary codes frozen, every scale
is trained as `α₀(1+θ)` to match the BF16 teacher's top-64 log-probabilities plus a rest-of-vocabulary
bucket, on 768 × 2048 tokens, then snapped to the u8 grid.

**Calibration.** Web/wiki 25%, English chat 20%, code 15%, multilingual chat (Aya, 20 languages) 25%,
math (MetaMathQA) 15%, so that the Hessians and the distillation targets cover the domains the model
serves.

**High precision kept** for the DeltaNet gates, conv1d and norms; the embedding (Q3_K) and LM head
(Q5_K) use llama.cpp k-quants with the Unsloth imatrix; the MTP block uses the Q4_K_M recipe.

## Reproduce

One CUDA GPU with ≥40 GB, ~80 GB of disk, Python 3.12, CUDA toolkit and CMake.

```bash
pip install -r requirements.txt          # pinned versions; see the file for the torch/CUDA wheel
source scripts/env.sh                    # paths default to ./work; override SAQ_WORK, SAQ_MODEL, ...
scripts/download.sh                      # Qwen/Qwen3.5-9B + unsloth/Qwen3.5-9B-MTP-GGUF baselines
scripts/setup_llama.sh                   # llama.cpp b11538 + llama.cpp/shiftadd-b11538.patch, CUDA build, kernel tests
python tools/make_eval_text.py           # wiki.test.raw, mix.test.raw
scripts/pipeline.sh 4.0                  # sensitivity -> plan -> quantize -> distill -> GGUF -> KLD / tasks / speed
scripts/eval_baselines.sh                # same measurements for the Unsloth quants
python tools/collect.py && python tools/plot.py
```

`FT_NSEQ=64 scripts/pipeline.sh 4.0` runs a short end-to-end check. The sensitivity step runs once and
is shared by all target sizes; its four shards (`--layers 0-7`, …) can run on separate GPUs.
`slurm/example.sbatch` wraps any step for a cluster.

## Layout

| path | contents |
|---|---|
| `saq/bcq.py`, `saq/gptq_bcq.py` | BCQ primitives; output-aware fit of one linear (row-wise scales) |
| `saq/colwise.py` | block-wise / column-wise scales: fit, exact refit, packing |
| `saq/quantize_model.py` | sequential quantization of all decoder linears |
| `saq/sens_kld.py`, `saq/sensitivity.py` | KLD sensitivity, knapsack planner |
| `saq/qlinear.py`, `saq/finetune.py` | trainable ShiftAdd linears, scale distillation |
| `saq/export_gguf.py` | GGUF writer for the new types (bit-exact packing and layout checks) |
| `saq/data.py`, `saq/model.py` | calibration mixes, Qwen3.5 helpers, PyTorch KLD |
| `llama.cpp/shiftadd-b11538.patch` | ggml types: CPU reference, CUDA dequant / get_rows / mat-vec, gguf-py, tests |
| `scripts/`, `tools/` | setup, pipeline, evaluation, benchmarks, plots |

## Citation

```bibtex
@article{you2024shiftaddllm,
  title={ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization},
  author={You, Haoran and Guo, Yipin and Fu, Yichao and Zhou, Wei and Shi, Huihong and Zhang, Xiaofan and Kundu, Souvik and Yazdanbakhsh, Amir and Lin, Yingyan},
  journal={arXiv preprint arXiv:2406.05981},
  year={2024}
}
```
