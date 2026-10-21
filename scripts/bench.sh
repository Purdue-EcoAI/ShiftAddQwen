#!/bin/bash
# Throughput + memory for one GGUF on this GPU.
#   scripts/bench.sh <model.gguf> <label> [port]
# 1) llama-bench: prefill pp512 and decode tg128 tok/s (flash-attn, full offload, 5 reps)
# 2) resident GPU memory of llama-server at an on-device assistant setting (65K ctx, 1 slot) after a warm request
# 3) decode tok/s with MTP speculative decoding (--spec-type draft-mtp --spec-draft-n-max 3)
set -e
source "$(dirname "$0")/env.sh"
MODEL=$1; LABEL=$2; PORT=${3:-8095}; BIN=$LLAMA_BIN
OUT=$SAQ_WORK/results/bench; mkdir -p $OUT
$BIN/llama-bench -m $MODEL -ngl 99 -fa 1 -p 512 -n 128 -r 5 -o json > $OUT/$LABEL.llama-bench.json 2> $OUT/$LABEL.llama-bench.err

measure_server () {  # $1 = extra flags, $2 = tag
  $BIN/llama-server -m $MODEL -ngl 99 -fa on -c 65536 -np 1 --jinja --host 127.0.0.1 --port $PORT --no-webui $1 \
      > $OUT/$LABEL.server$2.log 2>&1 &
  local SPID=$!
  for i in $(seq 1 300); do
    curl -s -o /dev/null -w "%{http_code}" 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break; sleep 1
  done
  python3 - $PORT $OUT/$LABEL.server$2.json <<'EOF'
import json, sys, time, urllib.request, subprocess
port, out = sys.argv[1], sys.argv[2]
prompts = ["Write a detailed, well-structured essay about the history of the printing press and its impact on science.",
           "Explain step by step how to implement a binary search tree in Python, with code for insert, delete and search.",
           "List 40 countries with their capitals and one notable fact about each, as a JSON array of objects."]
res = []
for p in prompts:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 512,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    t = r.get("timings", {})
    res.append({"predicted_per_second": t.get("predicted_per_second"), "predicted_n": t.get("predicted_n"),
                "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")})
mem = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout
json.dump({"requests": res, "nvidia_smi": mem}, open(out, "w"), indent=1)
print(json.dumps(res))
EOF
  # resident GPU memory of this server process
  python3 - $SPID $OUT/$LABEL.server$2.json <<'EOF'
import json, sys, subprocess
pid, out = sys.argv[1], sys.argv[2]
d = json.load(open(out))
for line in d["nvidia_smi"].strip().splitlines():
    p, m = [x.strip() for x in line.split(",")]
    if p == pid:
        d["gpu_mem_mib"] = int(m)
json.dump(d, open(out, "w"), indent=1)
print("gpu_mem_mib", d.get("gpu_mem_mib"))
EOF
  kill $SPID; wait $SPID 2>/dev/null || true
}
measure_server "" ""
measure_server "--spec-type draft-mtp --spec-draft-n-max 3" ".mtp" || echo "no MTP head: speculative run skipped"
grep -E "model buffer size|KV buffer|compute buffer|RS buffer" $OUT/$LABEL.server.log | head -12 > $OUT/$LABEL.buffers.txt || true
