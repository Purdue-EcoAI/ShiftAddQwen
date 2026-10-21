#!/bin/bash
# Agentic / reasoning comparison of one GGUF: reasoning_eval (AIME-25, MATH-500, GPQA-Diamond, IFEval),
# BFCL v3 tool calling, tau-bench (retail, airline).  All through llama-server with tool-call + thinking support.
#   tools/agentic/run_agentic.sh <model.gguf> <label> [port]
# Env: STAGE=reasoning|bfcl|tau|all (default all), AGENTIC_VENV (bfcl + tau-bench venv), USER_SIM_URL (OpenAI-compatible endpoint used as tau-bench's user
#      simulator; defaults to the same server), SEEDS (default 4), THINK_TOKENS (default 32768)
set -e
source "$(dirname "$0")/../../scripts/env.sh"
MODEL=$1; LABEL=$2; PORT=${3:-8200}
OUT=$SAQ_WORK/results/agentic/$LABEL; mkdir -p $OUT
$LLAMA_BIN/llama-server -m $MODEL -ngl 99 -fa on -c 262144 -np 8 --jinja --reasoning-format deepseek \
   --host 127.0.0.1 --port $PORT --no-webui --alias default > $OUT/server.log 2>&1 &
SPID=$!; trap "kill $SPID 2>/dev/null; wait $SPID 2>/dev/null" EXIT
for i in $(seq 1 300); do curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:$PORT/v1/models | grep -q 200 && break; sleep 1; done
URL=http://127.0.0.1:$PORT
STAGE=${STAGE:-all}
has () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# 1) reasoning + instruction following (thinking on, recommended sampling, several seeds)
has reasoning && $PYTHON $SAQ_ROOT/tools/agentic/reasoning_eval.py --url $URL --out $OUT/reasoning.json --label $LABEL \
   --think --seeds ${SEEDS:-4} --max-tokens ${THINK_TOKENS:-32768} --tasks ${REASONING_TASKS:-aime25,math500,gpqa,ifeval}

# 2) BFCL v3 (function calling: simple/multiple/parallel, live, multi-turn, relevance).  BFCL talks to an
#    existing OpenAI-compatible server when --skip-server-setup is given (LOCAL_SERVER_ENDPOINT/PORT); the model
#    name selects the prompt/tool-call handler (BFCL_MODEL, default: the Qwen3 FC handler).
source ${AGENTIC_VENV:-$SAQ_WORK/agentic_venv}/bin/activate
export LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT=$PORT REMOTE_OPENAI_BASE_URL=$URL/v1 REMOTE_OPENAI_API_KEY=none OPENAI_API_KEY=none
BM=${BFCL_MODEL:-Qwen/Qwen3-8B-FC}
has bfcl && (cd $OUT && bfcl generate --model $BM --skip-server-setup --test-category simple,multiple,parallel,parallel_multiple,live,multi_turn,irrelevance \
      --num-threads 8 --temperature 0.6 > bfcl_generate.log 2>&1 && bfcl evaluate --model $BM > bfcl_eval.log 2>&1) \
   && echo "bfcl done" || echo "bfcl failed (see $OUT/bfcl_*.log)"

# 3) tau-bench retail + airline (agent = this model; user simulator = USER_SIM_URL or this model)
#    (tau-bench uses litellm; the openai provider honours OPENAI_API_BASE.  Agent and user simulator share the
#    server unless USER_SIM_URL points at a stronger model.)
export OPENAI_API_BASE=$URL/v1 OPENAI_BASE_URL=$URL/v1
TAU=${TAU_BENCH_DIR:-$SAQ_WORK/agentic/tau-bench}
has tau && (cd $OUT && for env in retail airline; do
   python $TAU/run.py --agent-strategy tool-calling --env $env --model default --model-provider openai \
      --user-model default --user-model-provider openai --user-strategy llm --temperature 0.6 \
      --max-concurrency 4 --num-trials ${SEEDS:-4} --log-dir tau_$env > tau_$env.log 2>&1 || echo "tau-bench $env failed"; done)
echo AGENTIC_DONE $LABEL
