#!/bin/bash
# KLD (wiki + mixed-domain), task suite and speed for the stock llama.cpp quants.
set -e
source "$(dirname "$0")/env.sh"
for q in Q8_0 Q4_K_M UD-Q3_K_XL Q3_K_M UD-IQ3_XXS UD-Q2_K_XL UD-IQ2_M; do
  TEXT=wiki $SAQ_ROOT/scripts/kld.sh $G/Qwen3.5-9B-$q.gguf $q
  TEXT=mix  $SAQ_ROOT/scripts/kld.sh $G/Qwen3.5-9B-$q.gguf $q
  $SAQ_ROOT/scripts/suite.sh $G/Qwen3.5-9B-$q.gguf $q
  $SAQ_ROOT/scripts/bench.sh $G/Qwen3.5-9B-$q.gguf $q
done
