#!/bin/bash
#SBATCH --job-name=skyrmion_pd_merge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load stack/2024-06 python/3.12.8

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export MPLBACKEND=Agg
export SKYRMION_OUTPUT_ROOT="${SKYRMION_OUTPUT_ROOT:-$SCRATCH/skyrmion_runs}"

mkdir -p logs "$SKYRMION_OUTPUT_ROOT"

RUN_NAME="${RUN_NAME:-pd_L32_50x50_chunks}"

python3 phase_diagram.py \
  --merge-run-name "$RUN_NAME" \
  --output-root "$SKYRMION_OUTPUT_ROOT"
