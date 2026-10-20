#!/bin/bash
# PPL + KLD vs the BF16 model (512-token windows, scored second half, 40 chunks).
#   TEXT=wiki|mix scripts/kld.sh <model.gguf> <label> [chunks]
#   wiki = wikitext-2 test (tools: saq.data), mix = held-out mixed-domain text (tools/make_mix_text.py)
set -e
source "$(dirname "$0")/env.sh"
MODEL=$1; LABEL=$2; CHUNKS=${3:-40}; TEXT=${TEXT:-wiki}
REF=$SAQ_WORK/data/ref_bf16_${TEXT}_c${CHUNKS}.kld
OUT=$SAQ_WORK/results/kld; mkdir -p $OUT
if [ ! -f $REF ]; then
  $LLAMA_BIN/llama-perplexity -m $G/Qwen3.5-9B-BF16.gguf -f $SAQ_WORK/data/$TEXT.test.raw -c 512 -b 512 \
     --chunks $CHUNKS -ngl 99 -fa on --kl-divergence-base $REF > $OUT/ref_bf16_${TEXT}_c$CHUNKS.log 2>&1
fi
$LLAMA_BIN/llama-perplexity -m $MODEL -f $SAQ_WORK/data/$TEXT.test.raw -c 512 -b 512 --chunks $CHUNKS -ngl 99 -fa on \
   --kl-divergence-base $REF --kl-divergence > $OUT/$LABEL.$TEXT.c$CHUNKS.log 2>&1
$PYTHON - "$OUT/$LABEL.$TEXT.c$CHUNKS.log" "$LABEL" "$MODEL" <<'PY'
import json, re, sys, os
log = open(sys.argv[1]).read()
g = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, log))
r = dict(label=sys.argv[2], file_gb=os.path.getsize(sys.argv[3]) / 1e9,
         ppl=g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), ppl_base=g(r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
         kld=g(r"Mean\s+KLD:\s*([\d.]+)"), kld_p99=g(r"99\.0%\s+KLD:\s*([\d.]+)"), top1=g(r"Same top p:\s*([\d.]+)"))
print(json.dumps(r))
open(sys.argv[1].replace(".log", ".json"), "w").write(json.dumps(r))
PY
