#!/bin/bash
#SBATCH --job-name=merge_hdf5
#SBATCH --account=pi-haky
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
# --output and --error set by mergeHDF5.sh via -o / -e flags

set -euo pipefail

# ── Variables injected via --export from mergeHDF5.sh ────────────────────────
# Required:
#   MERGE_SCRIPT   path to merge_hdf5.py
#   H5_DIR         directory containing HDF5 files to merge
#   OUTPUT         output path (.h5 or directory for parquet)
#
# Optional (defaults shown):
#   MODE           auto-detected if unset
#   OUTPUT_TYPES   unset → all stored types
#   HAPLOTYPE      sum
#   FORMAT         hdf5
#   SAMPLES_FILE   unset → all samples in H5_DIR

echo "=============================="
echo "Job     : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "H5 dir  : ${H5_DIR}"
echo "Output  : ${OUTPUT}"
echo "Format  : ${FORMAT:-hdf5}"
echo "Start   : $(date)"
echo "=============================="

source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$(dirname "$OUTPUT")"

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --h5-dir  "$H5_DIR"
    --output  "$OUTPUT"
    --format  "${FORMAT:-hdf5}"
    --haplotype "${HAPLOTYPE:-sum}"
)

[[ -n "${MODE:-}"         ]] && ARGS+=(--mode "$MODE")
[[ -n "${OUTPUT_TYPES:-}" ]] && ARGS+=(--output-types $OUTPUT_TYPES)
[[ -n "${SAMPLES_FILE:-}" ]] && ARGS+=(--samples "$SAMPLES_FILE")

python "$MERGE_SCRIPT" "${ARGS[@]}"

echo "=============================="
echo "End     : $(date)"
echo "Output  : $(ls -lh "$OUTPUT" 2>/dev/null || ls -lh "$(dirname "$OUTPUT")" 2>/dev/null | tail -5)"
echo "=============================="
