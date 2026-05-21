#!/bin/bash
# Merge AlphaGenome HDF5 prediction files.
#
# Runs interactively by default; use --slurm to submit as a SLURM job.
# Mode is auto-detected (reference_shard*.h5 → reference, else personalized).
#
# Usage:
#   bash mergeHDF5.sh --h5-dir <dir> --output <path> [options]
#   bash mergeHDF5.sh --h5-dir <dir> --output <path> --slurm [options]
#
# Options:
#   --mode          {personalized,reference}            (auto-detected)
#   --output-types  CHIP_TF ATAC ...                   (default: all stored)
#   --haplotype     {haplotype1,haplotype2,sum,mean}   (default: sum)
#   --format        {hdf5,parquet}                     (default: hdf5)
#   --samples       /path/to/samples.txt               (optional subset)
#   --slurm                                            submit via sbatch
#   --mem           SLURM memory                       (default: 64G)
#   --time          SLURM time limit                   (default: 4:00:00)
#
# Examples:
#   # interactive — personalized → HDF5
#   bash mergeHDF5.sh \
#       --h5-dir predictions/run_concordance/h5 \
#       --output predictions/run_concordance/merged.h5
#
#   # interactive — reference shards → parquet
#   bash mergeHDF5.sh \
#       --h5-dir predictions/run_reference/h5 \
#       --output predictions/run_reference/reference.parquet \
#       --format parquet
#
#   # SLURM submission
#   bash mergeHDF5.sh \
#       --h5-dir predictions/run/h5 \
#       --output predictions/run/merged.h5 \
#       --slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="$SCRIPT_DIR/merge_hdf5.py"
SBATCH_SCRIPT="$SCRIPT_DIR/mergeHDF5.sbatch"

# ── Defaults ──────────────────────────────────────────────────────────────────
H5_DIR=""
OUTPUT=""
MODE=""
OUTPUT_TYPES=""
HAPLOTYPE="sum"
FORMAT="hdf5"
SAMPLES_FILE=""
SLURM=false
MEM="64G"
TIME="4:00:00"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --h5-dir)        H5_DIR=$2;        shift 2 ;;
        --output)        OUTPUT=$2;        shift 2 ;;
        --mode)          MODE=$2;          shift 2 ;;
        --output-types)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                OUTPUT_TYPES="$OUTPUT_TYPES $1"; shift
            done
            OUTPUT_TYPES="${OUTPUT_TYPES# }"
            ;;
        --haplotype)     HAPLOTYPE=$2;     shift 2 ;;
        --format)        FORMAT=$2;        shift 2 ;;
        --samples)       SAMPLES_FILE=$2;  shift 2 ;;
        --slurm)         SLURM=true;       shift   ;;
        --mem)           MEM=$2;           shift 2 ;;
        --time)          TIME=$2;          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$H5_DIR" || -z "$OUTPUT" ]]; then
    echo "Usage: $0 --h5-dir <dir> --output <path> [options]"
    exit 1
fi

LOG_DIR="$(dirname "$OUTPUT")/logs"
mkdir -p "$LOG_DIR"

# ── SLURM submission ──────────────────────────────────────────────────────────
if $SLURM; then
    EXPORT_VARS="MERGE_SCRIPT=$MERGE_SCRIPT"
    EXPORT_VARS+=",H5_DIR=$H5_DIR"
    EXPORT_VARS+=",OUTPUT=$OUTPUT"
    EXPORT_VARS+=",FORMAT=$FORMAT"
    EXPORT_VARS+=",HAPLOTYPE=$HAPLOTYPE"
    [[ -n "$MODE"         ]] && EXPORT_VARS+=",MODE=$MODE"
    [[ -n "$OUTPUT_TYPES" ]] && EXPORT_VARS+=",OUTPUT_TYPES=$OUTPUT_TYPES"
    [[ -n "$SAMPLES_FILE" ]] && EXPORT_VARS+=",SAMPLES_FILE=$SAMPLES_FILE"

    JOB_ID=$(sbatch \
        --mem="$MEM" \
        --time="$TIME" \
        --export="$EXPORT_VARS" \
        -o "$LOG_DIR/merge_%j.out" \
        -e "$LOG_DIR/merge_%j.err" \
        "$SBATCH_SCRIPT" \
        | awk '{print $NF}')

    echo "=============================="
    echo "Submitted : $JOB_ID"
    echo "H5 dir    : $H5_DIR"
    echo "Output    : $OUTPUT"
    echo "Format    : $FORMAT"
    echo "Logs      : $LOG_DIR/merge_${JOB_ID}.out"
    echo "Monitor   : squeue -j $JOB_ID"
    echo "=============================="

# ── Interactive ───────────────────────────────────────────────────────────────
else
    source /home/temi/miniconda3/etc/profile.d/conda.sh
    conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

    ARGS=(
        --h5-dir    "$H5_DIR"
        --output    "$OUTPUT"
        --format    "$FORMAT"
        --haplotype "$HAPLOTYPE"
    )
    [[ -n "$MODE"         ]] && ARGS+=(--mode "$MODE")
    [[ -n "$OUTPUT_TYPES" ]] && ARGS+=(--output-types $OUTPUT_TYPES)
    [[ -n "$SAMPLES_FILE" ]] && ARGS+=(--samples "$SAMPLES_FILE")

    python "$MERGE_SCRIPT" "${ARGS[@]}"
fi
