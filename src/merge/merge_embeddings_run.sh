#!/bin/bash
# Merge AlphaGenome embedding shard HDF5 files.
#
# Runs interactively by default; use --slurm to submit as a SLURM job.
#
# Usage
# ─────
#   bash merge_embeddings_run.sh \
#       --h5-dir  /path/to/shards \
#       --output  /path/to/merged_embeddings.h5 \
#       [--format hdf5|parquet] \
#       [--slurm] [--mem 32G] [--time 01:00:00]
#
# Examples
# ────────
#   # interactive (HDF5 output)
#   bash merge_embeddings_run.sh \
#       --h5-dir  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/h5 \
#       --output  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/merged_embeddings.h5
#
#   # interactive (parquet — writes merged_embeddings.embeddings_128bp.parquet)
#   bash merge_embeddings_run.sh \
#       --h5-dir  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/h5 \
#       --output  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/merged_embeddings \
#       --format  parquet
#
#   # SLURM submission
#   bash merge_embeddings_run.sh \
#       --h5-dir  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/h5 \
#       --output  /scratch/midway2/temi/alphagenome/embeddings/AR_Prostate/merged_embeddings.h5 \
#       --slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_embeddings.py"
SBATCH_SCRIPT="${SCRIPT_DIR}/merge_embeddings.sbatch"

# ── Defaults ──────────────────────────────────────────────────────────────────
H5_DIR=""
OUTPUT=""
FORMAT="hdf5"
SLURM=false
MEM="32G"
TIME="01:00:00"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --h5-dir)  H5_DIR=$2;  shift 2 ;;
        --output)  OUTPUT=$2;  shift 2 ;;
        --format)  FORMAT=$2;  shift 2 ;;
        --slurm)   SLURM=true; shift   ;;
        --mem)     MEM=$2;     shift 2 ;;
        --time)    TIME=$2;    shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep -v '^set' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1  (use --help for usage)"; exit 1 ;;
    esac
done

[[ -z "$H5_DIR" ]] && { echo "ERROR: --h5-dir is required"; exit 1; }
[[ -z "$OUTPUT" ]] && { echo "ERROR: --output is required";  exit 1; }
[[ ! -d "$H5_DIR" ]] && { echo "ERROR: directory not found: $H5_DIR"; exit 1; }

N_SHARDS=$(ls "${H5_DIR}"/reference_emb_shard*.h5 2>/dev/null | wc -l)
LOG_DIR="$(dirname "${OUTPUT}")/logs"
mkdir -p "${LOG_DIR}"

echo "=============================="
echo "H5 dir    : ${H5_DIR}"
echo "Shards    : ${N_SHARDS}"
echo "Output    : ${OUTPUT}"
echo "Format    : ${FORMAT}"
echo "=============================="

# ── SLURM submission ──────────────────────────────────────────────────────────
if $SLURM; then
    EXPORT_VARS="MERGE_SCRIPT=${MERGE_SCRIPT}"
    EXPORT_VARS+=",H5_DIR=${H5_DIR}"
    EXPORT_VARS+=",OUTPUT=${OUTPUT}"
    EXPORT_VARS+=",FORMAT=${FORMAT}"

    JOB_ID=$(sbatch \
        --mem="${MEM}" \
        --time="${TIME}" \
        --export="${EXPORT_VARS}" \
        -o "${LOG_DIR}/merge_%j.out" \
        -e "${LOG_DIR}/merge_%j.err" \
        "${SBATCH_SCRIPT}" \
        | awk '{print $NF}')

    echo ""
    echo "Submitted : ${JOB_ID}"
    echo "Monitor   : squeue -j ${JOB_ID}"
    echo "Logs      : tail -f ${LOG_DIR}/merge_${JOB_ID}.out"

# ── Interactive ───────────────────────────────────────────────────────────────
else
    source /home/temi/miniconda3/etc/profile.d/conda.sh
    conda activate /beagle3/haky/users/shared_software/TFXcan-pipeline-tools

    mkdir -p "$(dirname "${OUTPUT}")"

    python "${MERGE_SCRIPT}" \
        --h5-dir "${H5_DIR}" \
        --output "${OUTPUT}" \
        --format "${FORMAT}"
fi
