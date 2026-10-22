#!/bin/bash
# Agentic comparison of one GGUF through llama-server (tool calls + thinking): BFCL v4 (non-live, live, multi-turn),
# tau-bench (retail, airline) and, optionally, reasoning_eval.py (AIME-25, MATH-500, GPQA-Diamond, IFEval).
#   tools/agentic/setup_agentic.sh                            # once: benchmarks + venv
#   tools/agentic/run_agentic.sh <model.gguf> <label> [port]  # results in $SAQ_WORK/results/agentic/<label>
#   python tools/agentic/collect_agentic.py && python tools/agentic/make_agentic_doc.py   # tables -> docs/results.md
# Env: STAGE=reasoning|bfcl|tau|all (default all), REASONING_TASKS, BFCL_CATEGORIES, BFCL_RUN, TAU_ENVS, TAU_CONC, TAU_TIMEOUT,
#      TAU_TASK_IDS (tools/agentic/tau_missing.py), NP (server slots, default 8), AGENTIC_VENV (bfcl + tau-bench venv),
#      USER_SIM_URL (OpenAI-compatible endpoint used as tau-bench's user simulator; defaults to the same server),
#      SEEDS (default 4; tau-bench trials), THINK_TOKENS (default 32768)
set -e
source "$(dirname "$0")/../../scripts/env.sh"
MODEL=$1; LABEL=$2; PORT=${3:-8200}
OUT=$SAQ_WORK/results/agentic/$LABEL; mkdir -p $OUT
STAGE=${STAGE:-all}
$LLAMA_BIN/llama-server -m $MODEL -ngl 99 -fa on -c $((32768 * ${NP:-8})) -np ${NP:-8} --jinja --reasoning-format deepseek \
   --host 127.0.0.1 --port $PORT --no-webui --alias default > $OUT/server_${STAGE:-all}_${SLURM_JOB_ID:-$$}.log 2>&1 &
SPID=$!; trap "kill $SPID 2>/dev/null; wait $SPID 2>/dev/null" EXIT
for i in $(seq 1 300); do curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:$PORT/v1/models | grep -q 200 && break; sleep 1; done
URL=http://127.0.0.1:$PORT
STAGE=${STAGE:-all}
has () { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }

# 1) reasoning + instruction following (thinking on, recommended sampling, several seeds)
has reasoning && $PYTHON $SAQ_ROOT/tools/agentic/reasoning_eval.py --url $URL --out $OUT/reasoning_${REASONING_TASKS:-all}.json --label $LABEL \
   --think --seeds ${SEEDS:-4} --max-tokens ${THINK_TOKENS:-32768} --tasks ${REASONING_TASKS:-aime25,math500,gpqa,ifeval}

# 2) BFCL (function calling: simple/multiple/parallel, live, multi-turn, relevance).  BFCL talks to the running
#    llama-server through the OpenAI chat-completions API with `tools` (model entry "llama-server-FC", registered by
#    tools/agentic/bfcl_local_model.py), so the GGUF's own chat template (Qwen3.5 XML tool calls) and llama.cpp's
#    tool-call parser are used; BFCL_MODEL overrides the entry.  Results/scores go to $OUT/bfcl (BFCL_PROJECT_ROOT).
source ${AGENTIC_VENV:-$SAQ_WORK/agentic_venv}/bin/activate
export OPENAI_BASE_URL=$URL/v1 OPENAI_API_KEY=none LOCAL_SERVER_ENDPOINT=127.0.0.1 LOCAL_SERVER_PORT=$PORT \
       REMOTE_OPENAI_BASE_URL=$URL/v1 REMOTE_OPENAI_API_KEY=none BFCL_PROJECT_ROOT=$OUT/bfcl${BFCL_RUN:-}   # BFCL_RUN=2: a repeat run (bfcl2/)
BM=${BFCL_MODEL:-llama-server-FC}
CATS=${BFCL_CATEGORIES:-simple,multiple,parallel,parallel_multiple,live,multi_turn,irrelevance}
has bfcl && (mkdir -p $BFCL_PROJECT_ROOT && cd $OUT && bfcl generate --model $BM --skip-server-setup --test-category $CATS \
      --num-threads 8 --temperature 0.6 > bfcl${BFCL_RUN:-}_generate_${CATS//,/_}.log 2>&1 && bfcl evaluate --model $BM --test-category $CATS > bfcl${BFCL_RUN:-}_eval_${CATS//,/_}.log 2>&1) \
   && echo "bfcl done" || echo "bfcl failed (see $OUT/bfcl_*.log)"

# 3) tau-bench retail + airline (agent = this model; user simulator = USER_SIM_URL or this model)
#    (tau-bench uses litellm; the openai provider honours OPENAI_API_BASE.  Agent and user simulator share the
#    server unless USER_SIM_URL points at a stronger model; the simulator runs without thinking, tools/agentic/tau_user_nothink.py.)
export OPENAI_API_BASE=$URL/v1 OPENAI_BASE_URL=$URL/v1 REQUEST_TIMEOUT=${TAU_TIMEOUT:-7200}   # litellm per-request timeout (s)
TAU_EXTRA=${TAU_TASK_IDS:+--task-ids $TAU_TASK_IDS}   # repair runs: only these task ids (tools/agentic/tau_missing.py)
TAU=${TAU_BENCH_DIR:-$SAQ_WORK/agentic/tau-bench}
has tau && (cd $OUT && for env in ${TAU_ENVS:-retail airline}; do
   python $TAU/run.py --agent-strategy tool-calling --env $env --model default --model-provider openai \
      --user-model default --user-model-provider openai --user-strategy llm --temperature 0.6 \
      --max-concurrency ${TAU_CONC:-${NP:-8}} --num-trials ${SEEDS:-4} --log-dir tau_$env $TAU_EXTRA > tau_$env.log 2>&1 || echo "tau-bench $env failed"; done)
echo AGENTIC_DONE $LABEL
