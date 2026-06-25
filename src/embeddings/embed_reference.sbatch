#!/bin/bash
#SBATCH --job-name=emb_reference
#SBATCH --account=beagle3-exusers
#SBATCH --partition=beagle3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
# --job-name, --time, --output, --error are overridden by extract_embeddings.sh

# ── Variables injected via --export ──────────────────────────────────────────
#   Required:
#     PREDICT_SCRIPT  — absolute path to predict_embeddings.py
#     BED_FILE        — intervals BED/TSV file
#     OUTPUT_DIR      — where shard HDF5s are written
#     N_SHARDS        — total number of shards (= array size)
#   Optional (have defaults):
#     FASTA_FILE      — genome FASTA (default: hg38 path in predict_embeddings.py)
#     AGGREGATE       — mean | max | none          (default: mean)
#     OUTPUT_TYPES    — embeddings_128bp[,embeddings_1bp]  (default: embeddings_128bp)

set -euo pipefail

# ── Defaults for optional exports ─────────────────────────────────────────────
AGGREGATE="${AGGREGATE:-mean}"
BINS="${BINS:-}"
OUTPUT_TYPES="${OUTPUT_TYPES:-embeddings_128bp}"
FASTA_FILE="${FASTA_FILE:-}"   # empty → predict_embeddings.py uses its hg38 default

# Build optional flags
FASTA_ARG=""
[[ -n "$FASTA_FILE" ]] && FASTA_ARG="--fasta-file ${FASTA_FILE}"
BINS_ARG=""
[[ -n "$BINS" ]] && BINS_ARG="--bins ${BINS}"

echo "=============================="
echo "Job         : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Shard       : ${SLURM_ARRAY_TASK_ID:-0} / ${N_SHARDS}"
echo "BED file    : ${BED_FILE}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "FASTA       : ${FASTA_FILE:-<default hg38>}"
echo "Aggregate   : ${AGGREGATE}"
echo "Bins        : ${BINS:-all}"
echo "Types       : ${OUTPUT_TYPES}"
echo "Script      : ${PREDICT_SCRIPT}"
echo "Start       : $(date)"
echo "=============================="

module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "${OUTPUT_DIR}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 python "${PREDICT_SCRIPT}" \
    --bed-file     "${BED_FILE}" \
    --output-dir   "${OUTPUT_DIR}" \
    --n-shards     "${N_SHARDS}" \
    --aggregate    "${AGGREGATE}" \
    --output-types "${OUTPUT_TYPES}" \
    ${FASTA_ARG} \
    ${BINS_ARG}

echo "=============================="
echo "End         : $(date)"
N_H5=$(ls "${OUTPUT_DIR}"/reference_emb_shard*.h5 2>/dev/null | wc -l)
echo "Shards done : ${N_H5} / ${N_SHARDS}"
echo "=============================="
