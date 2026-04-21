#!/bin/bash
#SBATCH --job-name=skyrmion_pd
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2000M
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load stack/2024-06 python/3.12.8

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export SKYRMION_OUTPUT_ROOT="${SKYRMION_OUTPUT_ROOT:-$SCRATCH/skyrmion_runs}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba_cache}"

mkdir -p logs "$SKYRMION_OUTPUT_ROOT" "$NUMBA_CACHE_DIR"

RUN_NAME="${RUN_NAME:-pd_L32_50x50}"
NH="${NH:-50}"
NA="${NA:-50}"
LATTICE_SIZE="${LATTICE_SIZE:-32}"
H_MIN="${H_MIN:-0.0}"
H_MAX="${H_MAX:-2.5}"
A_MIN="${A_MIN:--1.5}"
A_MAX="${A_MAX:-1.7}"

srun --cpu-bind=cores python3 -u phase_diagram.py \
  --nH "$NH" \
  --nA "$NA" \
  --L "$LATTICE_SIZE" \
  --H-min "$H_MIN" \
  --H-max "$H_MAX" \
  --A-min "$A_MIN" \
  --A-max "$A_MAX" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \
  --run-name "$RUN_NAME" \
  --output-root "$SKYRMION_OUTPUT_ROOT"
