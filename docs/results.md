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

<!-- agentic:start -->
## Agentic benchmarks: tool calling

| model | GGUF (GB) | KLD mixed ↓ | prefill tok/s ↑ | decode tok/s ↑ | BFCL non-live ↑ | BFCL live ↑ | BFCL multi-turn ↑ | τ-bench retail pass^1 ↑ | τ-bench airline pass^1 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| UnslothAI/Qwen-3.5-9B-Q4_K_M | 5.87 | 0.0257 | 4227 | 125.2 | 86.9 | 79.7 | 46.9 (47.0 / 46.9) | 47.4 (230 ep.) | 55.0 (100 ep.) |
| **ShiftAddQwen-3.5-9B-4.4bits** | 5.10 (-13%) | 0.0303 | 5525 (+31%) | 129.9 (+4%) | 85.9 ≈-1.1 | 79.9 ≈+1.8 | 43.8 (43.6 / 43.9) ▼-3.2 | 51.3 (230 ep.) ▲+3.9 | 53.0 (100 ep.) ≈-2.0 |
| UnslothAI/Qwen-3.5-9B-Q3_K_M | 4.83 (-18%) | 0.0669 | 3529 (-16%) | 111.0 (-11%) | 86.7 ≈-1.7 | 79.2 ≈-0.4 | 43.4 (43.6 / 43.1) ▼-3.6 | 50.4 (230 ep.) ▲+3.0 | 46.0 (100 ep.) ▼-9.0 |
| **ShiftAddQwen-3.5-9B-4.0bits** | 4.76 (-19%) | 0.0384 | 5367 (+27%) | 134.6 (+8%) | 84.8 ≈-1.8 | 79.3 ≈+1.9 | 43.6 (43.2 / 43.9) ▼-3.4 | 48.7 (230 ep.) ≈+1.3 | 50.0 (100 ep.) ▼-5.0 |

*Table 2. Tool calling of the two drop-in candidates against the reference 4-bit deployment (and Q3_K_M, the standard 3-bit recipe of similar size), all served
by the same patched `llama-server` (`--jinja`, thinking on, temperature 0.6 / top-p 0.95). GGUF size, KLD to BF16 on the
mixed-domain text and `llama-bench` prefill (pp512) / decode (tg128) throughput on the A100 are repeated from Table 1, with
the change relative to Q4_K_M in parentheses. BFCL v4 (Berkeley Function
Calling Leaderboard): accuracy (%) of the non-live (1,390 items), live (2,251) and multi-turn (800) splits, scored by
`bfcl evaluate`; multi-turn is the mean of two generation runs (both values in parentheses). τ-bench: pass^1 (%) = mean task success over 2 trials of the retail (115 tasks) and airline (50 tasks)
environments; the user simulator is the same model with thinking off. ▲ / ▼ / ≈ after a score: better / worse than / within 2 points of
Q4_K_M, followed by the per-item difference in points (paired over the same BFCL items and τ-bench tasks;
it can differ from the difference of the BFCL overall numbers, which are BFCL's category-weighted means). Raw numbers and paired bootstrap 95% intervals: [`results/agentic.json`](../results/agentic.json).*

**BFCL per category** (accuracy, %):

| model | simple (py) | simple (java) | simple (js) | multiple | parallel | par. multiple | irrelevance | live simple | live multiple | live parallel | live par. mult. | live irrelevance | live relevance | MT base | MT miss func | MT miss param | MT long ctx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UnslothAI/Qwen-3.5-9B-Q4_K_M | 95.2 | 59.0 | 56.0 | 96.0 | 91.5 | 90.0 | 86.2 | 86.4 | 78.0 | 93.8 | 75.0 | 77.5 | 87.5 | 70.5 | 30.0 | 45.5 | 41.8 |
| **ShiftAddQwen-3.5-9B-4.4bits** | 94.2 | 58.0 | 54.0 | 95.0 | 90.5 | 89.5 | 84.6 | 81.4 | 79.5 | 87.5 | 79.2 | 81.7 | 87.5 | 62.8 | 27.5 | 46.8 | 38.0 |
| UnslothAI/Qwen-3.5-9B-Q3_K_M | 94.2 | 61.0 | 56.0 | 95.5 | 91.0 | 90.0 | 78.3 | 82.6 | 78.2 | 87.5 | 83.3 | 77.2 | 87.5 | 65.2 | 22.5 | 43.0 | 42.8 |
| **ShiftAddQwen-3.5-9B-4.0bits** | 93.5 | 58.0 | 52.0 | 92.0 | 91.5 | 88.0 | 85.0 | 82.2 | 78.5 | 93.8 | 75.0 | 82.8 | 93.8 | 63.8 | 28.8 | 43.5 | 38.2 |

**τ-bench** (pass^k = fraction of tasks solved in all k trials):

| model | retail pass^1 | retail pass^2 | retail episodes | airline pass^1 | airline pass^2 | airline episodes |
|---|---|---|---|---|---|---|
| UnslothAI/Qwen-3.5-9B-Q4_K_M | 47.4 | 33.0 | 230 | 55.0 | 46.0 | 100 |
| **ShiftAddQwen-3.5-9B-4.4bits** | 51.3 | 32.2 | 230 | 53.0 | 44.0 | 100 |
| UnslothAI/Qwen-3.5-9B-Q3_K_M | 50.4 | 34.8 | 230 | 46.0 | 34.0 | 100 |
| **ShiftAddQwen-3.5-9B-4.0bits** | 48.7 | 32.2 | 230 | 50.0 | 40.0 | 100 |

**Protocol.** `tools/agentic/run_agentic.sh` starts `llama-server` (8 parallel slots for BFCL, 16 for τ-bench; 32K context each) and runs BFCL (`bfcl generate` through the OpenAI chat-completions API with
`tools`, so the GGUF's own Qwen3.5 chat template and llama.cpp's XML tool-call parser are exercised; model entry
`llama-server-FC` from `tools/agentic/bfcl_local_model.py`) and τ-bench (`tool-calling` agent through litellm; user simulator patched by `tools/agentic/tau_user_nothink.py`). Hardware:
one NVIDIA V100-SXM2-32GB per (model, split). Sampling noise: BFCL splits are one generation per item at temperature 0.6,
so differences of about 1 point (non-live/live) to 2 points (multi-turn) are within run-to-run noise; τ-bench pass^1 over
230 / 100 episodes has a standard error of roughly 3 / 5 points.
<!-- agentic:end -->
