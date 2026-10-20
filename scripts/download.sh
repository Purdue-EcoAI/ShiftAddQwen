#!/bin/bash
# bf16 checkpoint + the reference / baseline GGUFs (Unsloth's MTP-head builds of Qwen3.5-9B)
set -e
source "$(dirname "$0")/env.sh"
hf download Qwen/Qwen3.5-9B --local-dir $SAQ_MODEL
hf download unsloth/Qwen3.5-9B-MTP-GGUF \
  "Qwen3.5-9B-BF16.gguf" "Qwen3.5-9B-Q8_0.gguf" "Qwen3.5-9B-Q4_K_M.gguf" "Qwen3.5-9B-Q3_K_M.gguf" \
  "Qwen3.5-9B-UD-Q3_K_XL.gguf" "Qwen3.5-9B-UD-IQ3_XXS.gguf" "Qwen3.5-9B-UD-Q2_K_XL.gguf" "Qwen3.5-9B-UD-IQ2_M.gguf" \
  "imatrix_unsloth.gguf_file" --local-dir $G
