#!/bin/bash
# Clone llama.cpp at b11538, apply the ShiftAdd patch, build with CUDA (set CUDA_ARCH, default 80 = A100).
set -e
source "$(dirname "$0")/env.sh"
[ -d $LLAMA_CPP ] || git clone --depth 1 --branch b11538 https://github.com/ggml-org/llama.cpp.git $LLAMA_CPP
cd $LLAMA_CPP
git apply --check $SAQ_ROOT/llama.cpp/shiftadd-b11538.patch 2>/dev/null && git apply $SAQ_ROOT/llama.cpp/shiftadd-b11538.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-80} -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
      -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j ${JOBS:-16} --target llama-perplexity llama-bench llama-quantize llama-server test-backend-ops
./build/bin/test-backend-ops -o MUL_MAT -b CUDA0 | tail -3
