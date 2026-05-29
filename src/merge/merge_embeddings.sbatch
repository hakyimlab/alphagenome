#!/bin/bash
#SBATCH --job-name=merge_embeddings
#SBATCH --account=pi-haky
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# --mem, --time, --output, --error are overridden by merge_embeddings_run.sh

set -euo pipefail

# ── Variables injected via --export ──────────────────────────────────────────
#   Required:
#     MERGE_SCRIPT  — path to merge_embeddings.py
#     H5_DIR        — directory containing reference_emb_shard*.h5 files
#     OUTPUT        — output path (.h5 or stem for parquet)
#   Optional (have defaults):
#     FORMAT        — hdf5 | parquet  (default: hdf5)

FORMAT="${FORMAT:-hdf5}"

echo "=============================="
echo "Job     : ${SLURM_JOB_ID}"
echo "Node    : ${SLURMD_NODENAME}"
echo "H5 dir  : ${H5_DIR}"
echo "Output  : ${OUTPUT}"
echo "Format  : ${FORMAT}"
echo "Start   : $(date)"
echo "=============================="

source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$(dirname "${OUTPUT}")"

python "${MERGE_SCRIPT}" \
    --h5-dir "${H5_DIR}" \
    --output "${OUTPUT}" \
    --format "${FORMAT}"

echo "=============================="
echo "End     : $(date)"
echo "Output  : $(ls -lh "${OUTPUT}" 2>/dev/null || echo '(parquet — see output dir)')"
echo "=============================="
