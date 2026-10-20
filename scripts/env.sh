# Source this from every script. Override any variable in your shell before running.
export SAQ_ROOT=${SAQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SAQ_WORK=${SAQ_WORK:-$SAQ_ROOT/work}                 # models, GGUFs, runs, results (tens of GB)
export SAQ_MODEL=${SAQ_MODEL:-$SAQ_WORK/models/Qwen3.5-9B}  # HF checkpoint (bf16)
export LLAMA_CPP=${LLAMA_CPP:-$SAQ_WORK/llama.cpp}          # patched llama.cpp source tree
export LLAMA_BIN=${LLAMA_BIN:-$LLAMA_CPP/build/bin}         # its binaries (llama-perplexity, llama-server, ...)
export PYTHON=${PYTHON:-python}
export PYTHONPATH=$SAQ_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLAMA_BIN:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
G=$SAQ_WORK/gguf/mtp      # downloaded reference/baseline GGUFs
O=$SAQ_WORK/gguf/ours     # exported ShiftAdd GGUFs
