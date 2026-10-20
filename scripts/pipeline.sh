#!/bin/bash
# Full recipe for one ShiftAdd model at a target average bpw over the decoder linears.
#   scripts/pipeline.sh <avg_bpw> [steps]      e.g. scripts/pipeline.sh 4.0
# steps (comma list, default all): sens,plan,quant,ft,export,kld,suite,bench
# One A100-40GB is enough. Rough times on A100: sens 4x~45 min (shardable), quant ~45 min, ft ~40-80 min.
set -e
source "$(dirname "$0")/env.sh"
B=$1; STEPS=${2:-sens,plan,quant,ft,export,kld,suite,bench}; N=${RUN_NAME:-v2g$B}
has () { [[ ,$STEPS, == *,$1,* ]]; }
cd $SAQ_ROOT
if has sens && [ $(ls $SAQ_WORK/results/sens_kld/L*.json 2>/dev/null | wc -l) -lt 4 ]; then
  for r in 0-7 8-15 16-23 24-31; do $PYTHON -m saq.sens_kld --layers $r; done   # one shard per GPU if you have several
fi
has plan && $PYTHON -m saq.sensitivity plan $B kldG
has quant && $PYTHON -m saq.quantize_model --out $SAQ_WORK/runs/$N --plan $SAQ_WORK/plans/plan_kldG_$B.json \
                 --act-order --true-seq --calib 2
has ft && $PYTHON -m saq.finetune --run $SAQ_WORK/runs/$N --out $SAQ_WORK/runs/${N}_ft --nseq ${FT_NSEQ:-768} --lr 1e-4 --calib 2
if has export; then
  [ -f $O/base_e3o5.gguf ] || $SAQ_ROOT/scripts/make_base.sh
  $PYTHON -m saq.export_gguf --run $SAQ_WORK/runs/${N}_ft --base $O/base_e3o5.gguf --ref $G/Qwen3.5-9B-BF16.gguf \
          --out $O/${N}_ft.gguf
fi
has kld && { TEXT=wiki $SAQ_ROOT/scripts/kld.sh $O/${N}_ft.gguf ${N}_ft; TEXT=mix $SAQ_ROOT/scripts/kld.sh $O/${N}_ft.gguf ${N}_ft; }
has suite && $SAQ_ROOT/scripts/suite.sh $O/${N}_ft.gguf ${N}_ft
has bench && $SAQ_ROOT/scripts/bench.sh $O/${N}_ft.gguf ${N}_ft
echo "done: $O/${N}_ft.gguf"
