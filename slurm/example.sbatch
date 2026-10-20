#!/bin/bash
# Example Slurm wrapper; adjust account/partition/QoS for your cluster.
#SBATCH -A <account>
#SBATCH -p <gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -J shiftadd
#SBATCH -o shiftadd_%j.log
# usage: sbatch slurm/example.sbatch scripts/pipeline.sh 4.0 quant,ft
# Tip: Python imports from a parallel filesystem can be very slow; put the venv on node-local disk.
cd "$SLURM_SUBMIT_DIR"
"$@"
