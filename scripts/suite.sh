#!/bin/bash
# Task suite for one GGUF:  scripts/suite.sh <model.gguf> <label> [port]
set -e
source "$(dirname "$0")/env.sh"
MODEL=$1; LABEL=$2; PORT=${3:-8091}
OUT=$SAQ_WORK/results/suite; mkdir -p $OUT
[ -f $SAQ_WORK/data/suite.json ] || $PYTHON $SAQ_ROOT/tools/eval/build_suite.py $SAQ_WORK/data/suite.json
$LLAMA_BIN/llama-server -m $MODEL -ngl 99 -fa on -c 40960 -np 4 --kv-unified --host 127.0.0.1 --port $PORT \
   --no-webui > $OUT/server_$LABEL.log 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null; wait $SPID 2>/dev/null" EXIT
for i in $(seq 1 300); do
  curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -20 $OUT/server_$LABEL.log; exit 1; }
  sleep 1
done
$PYTHON $SAQ_ROOT/tools/eval/run_eval.py --url http://127.0.0.1:$PORT --suite $SAQ_WORK/data/suite.json \
   --out $OUT/$LABEL.suite.json --label $LABEL
