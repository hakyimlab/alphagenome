#!/bin/bash
#SBATCH --job-name=emb_personalized
#SBATCH --account=beagle3-exusers
#SBATCH --partition=beagle3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
# --job-name, --time, --output, --error are overridden by extract_embeddings.sh

# ── Variables injected via --export ──────────────────────────────────────────
#   Required:
#     PREDICT_SCRIPT  — absolute path to predict_embeddings.py
#     BED_FILE        — intervals BED/TSV file
#     SAMPLES_FILE    — one sample ID per line
#     VCF_DIR         — directory containing per-chromosome VCF files
#     VCF_PATTERN     — filename pattern with {chrom} placeholder
#                       e.g. ALL.{chrom}.shapeit2_integrated_SNPs_v2a.GRCh38.phased.vcf.gz
#     OUTPUT_DIR      — where per-sample HDF5s are written
#     N_SHARDS        — total number of array tasks (= number of sample chunks)
#   Optional (have defaults):
#     FASTA_FILE      — genome FASTA (default: hg38 path in predict_embeddings.py)
#     AGGREGATE       — mean | max | none          (default: mean)
#     OUTPUT_TYPES    — embeddings_128bp[,embeddings_1bp]  (default: embeddings_128bp)

set -euo pipefail

AGGREGATE="${AGGREGATE:-mean}"
OUTPUT_TYPES="${OUTPUT_TYPES:-embeddings_128bp}"
FASTA_FILE="${FASTA_FILE:-}"

FASTA_ARG=""
[[ -n "$FASTA_FILE" ]] && FASTA_ARG="--fasta-file ${FASTA_FILE}"

N_SAMPLES=$(wc -l < "${SAMPLES_FILE}")

echo "=============================="
echo "Job         : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Node        : ${SLURMD_NODENAME:-$(hostname)}"
echo "Shard       : ${SLURM_ARRAY_TASK_ID:-0} / ${N_SHARDS}"
echo "Samples     : ${N_SAMPLES} total (this task gets ~$((N_SAMPLES / N_SHARDS)))"
echo "BED file    : ${BED_FILE}"
echo "VCF dir     : ${VCF_DIR}"
echo "VCF pattern : ${VCF_PATTERN}"
echo "FASTA       : ${FASTA_FILE:-<default hg38>}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Aggregate   : ${AGGREGATE}"
echo "Types       : ${OUTPUT_TYPES}"
echo "Start       : $(date)"
echo "=============================="

module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "${OUTPUT_DIR}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 python "${PREDICT_SCRIPT}" \
    --mode         personalized \
    --bed-file     "${BED_FILE}" \
    --samples-file "${SAMPLES_FILE}" \
    --vcf-dir      "${VCF_DIR}" \
    --vcf-pattern  "${VCF_PATTERN}" \
    --output-dir   "${OUTPUT_DIR}" \
    --n-shards     "${N_SHARDS}" \
    --aggregate    "${AGGREGATE}" \
    --output-types "${OUTPUT_TYPES}" \
    ${FASTA_ARG}

echo "=============================="
echo "End         : $(date)"
N_H5=$(ls "${OUTPUT_DIR}"/*_emb.h5 2>/dev/null | wc -l)
echo "HDF5 done   : ${N_H5} samples"
echo "=============================="
