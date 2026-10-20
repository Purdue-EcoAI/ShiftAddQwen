#!/bin/bash
# Base GGUF that supplies every non-BCQ tensor (embedding Q3_K, LM head Q5_K, MTP block, gates, norms).
# The exporter then swaps the decoder linears for SA tensors.
set -e
source "$(dirname "$0")/env.sh"
mkdir -p $O
$LLAMA_BIN/llama-quantize --imatrix $G/imatrix_unsloth.gguf_file --token-embedding-type q3_K \
  --output-tensor-type q5_K $G/Qwen3.5-9B-BF16.gguf $O/base_e3o5.gguf Q4_K_M ${JOBS:-8}
