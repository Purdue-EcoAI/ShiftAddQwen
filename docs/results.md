# Full results

| model | GGUF (GB) | KLD mixed ↓ | KLD wiki ↓ | PPL wiki ↓ | tasks ↑ | GSM8K | MMLU-Pro | HumanEval | MGSM | decode tok/s ↑ | prefill tok/s ↑ | decode+MTP tok/s ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UnslothAI/Qwen-3.5-9B-Q8_0 | 9.79 | 0.0008 | 0.0013 | 7.074 | 77.9 | 94.0 | 51.7 | 84.8 | 81.2 | 107.0 | 4056 | 169.4 |
| UnslothAI/Qwen-3.5-9B-Q4_K_M | 5.87 | 0.0257 | 0.0289 | 7.109 | 79.2 | 94.4 | 52.6 | 87.2 | 82.5 | 125.2 | 4227 | 158.2 |
| UnslothAI/Qwen-3.5-9B-UD-Q3_K_XL | 5.24 | 0.0362 | 0.0447 | 7.274 | 78.9 | 94.0 | 54.9 | 82.9 | 83.8 | 115.8 | 3789 | 141.0 |
| **ShiftAddQwen-3.5-9B-4.4bits** | 5.10 | 0.0303 | 0.0351 | 7.195 | 78.2 | 93.6 | 51.4 | 86.6 | 81.2 | 129.9 | 5525 | 183.3 |
| UnslothAI/Qwen-3.5-9B-Q3_K_M | 4.83 | 0.0669 | 0.0787 | 7.206 | 76.2 | 92.4 | 50.3 | 84.8 | 77.5 | 111.0 | 3529 | 140.7 |
| **ShiftAddQwen-3.5-9B-4.0bits** | 4.76 | 0.0384 | 0.0458 | 7.259 | 77.4 | 92.4 | 49.7 | 84.8 | 82.5 | 134.6 | 5367 | 171.5 |
| UnslothAI/Qwen-3.5-9B-UD-Q2_K_XL | 4.44 | 0.1369 | 0.1564 | 7.940 | 72.8 | 92.4 | 47.4 | 77.4 | 73.8 | 126.1 | 3288 | 140.1 |
| **ShiftAddQwen-3.5-9B-3.6bits** | 4.41 | 0.0564 | 0.0655 | 7.325 | 75.3 | 94.0 | 48.6 | 82.3 | 76.2 | 137.4 | 5344 | 173.6 |
| UnslothAI/Qwen-3.5-9B-UD-IQ3_XXS | 4.30 | 0.0880 | 0.0979 | 7.502 | 74.8 | 92.4 | 46.9 | 81.1 | 78.8 | 133.6 | 4120 | 166.9 |
| **ShiftAddQwen-3.5-9B-3.2bits** | 4.07 | 0.0857 | 0.0974 | 7.539 | 74.8 | 91.2 | 50.0 | 80.5 | 77.5 | 142.7 | 5319 | 177.0 |
| UnslothAI/Qwen-3.5-9B-UD-IQ2_M | 3.97 | 0.1501 | 0.1649 | 7.921 | 70.7 | 90.0 | 42.3 | 78.0 | 72.5 | 135.4 | 3415 | 157.1 |
| **ShiftAddQwen-3.5-9B-3.0bits** | 3.90 | 0.1052 | 0.1230 | 7.678 | 73.2 | 88.8 | 49.7 | 80.5 | 73.8 | 144.8 | 5342 | 175.4 |
| **ShiftAddQwen-3.5-9B-2.6bits** | 3.55 | 0.1701 | 0.1973 | 8.204 | 68.0 | 84.8 | 42.3 | 71.3 | 73.8 | 151.0 | 5379 | 180.9 |

*Table 1. Full measurements; [`results/summary.json`](../results/summary.json) holds the raw numbers. "tasks" is the mean of the
four task columns.*

**Protocol.** KLD and PPL: `llama-perplexity` against the BF16 GGUF, 512-token windows, 40 chunks, on
wikitext-2 test ("wiki") and on held-out web / chat / code / multilingual / math text ("mixed",
`tools/make_mix_text.py`). Tasks: greedy decoding, thinking disabled, via `llama-server`, on GSM8K
(250), MMLU-Pro (350), HumanEval (164) and MGSM (80) items (`tools/eval`); one run per model, so
differences below ~1.5 points are noise (Q8_0 scores below Q4_K_M). Speed: NVIDIA A100-PCIE-40GB,
`llama-bench` pp512 / tg128, and `llama-server` at 65K context, with and without
`--spec-type draft-mtp --spec-draft-n-max 3`.
