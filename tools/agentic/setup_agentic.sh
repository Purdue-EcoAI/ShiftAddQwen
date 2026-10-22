#!/bin/bash
# One-time setup for the agentic benchmarks (BFCL v4 + tau-bench) against a llama-server endpoint.
# Creates $SAQ_WORK/agentic_venv (bfcl_eval, tau_bench, litellm) and clones the two benchmarks into $SAQ_WORK/agentic.
set -e
source "$(dirname "$0")/../../scripts/env.sh"
A=$SAQ_WORK/agentic; mkdir -p $A
[ -d $A/gorilla ] || git clone --depth 1 https://github.com/ShishirPatil/gorilla.git $A/gorilla
[ -d $A/tau-bench ] || git clone --depth 1 https://github.com/sierra-research/tau-bench.git $A/tau-bench
V=$SAQ_WORK/agentic_venv
[ -x $V/bin/python ] || python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu      # bfcl_eval pulls sentence-transformers
$V/bin/pip install -q -e $A/gorilla/berkeley-function-call-leaderboard -e $A/tau-bench soundfile mcp
$PYTHON $SAQ_ROOT/tools/agentic/bfcl_local_model.py $A/gorilla/berkeley-function-call-leaderboard   # "llama-server-FC" model entry
$PYTHON $SAQ_ROOT/tools/agentic/tau_user_nothink.py $A/tau-bench      # user simulator without thinking
$PYTHON $SAQ_ROOT/tools/agentic/tau_ckpt_pid.py $A/tau-bench          # unique checkpoint file per process
# reasoning_eval.py (AIME etc.) runs in the main venv: pip install datasets requests nltk langdetect immutabledict
echo "agentic setup done: $V"
