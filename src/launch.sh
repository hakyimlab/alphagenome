#!/bin/bash
# Submit one SLURM job per GPU chunk.
#
# Usage:
#   bash launch.sh [--n-gpus N] [--samples file] [--bed file] [--outdir dir] [--no-profile]
#
# Example:
#   bash launch.sh --n-gpus 20
#   bash launch.sh --n-gpus 10 --samples files/my_samples.tsv

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
N_GPUS=8
SCRIPT=/beagle3/haky/users/temi/projects/alphagenome/src/predict.py
BED=/beagle3/haky/users/temi/projects/alphagenome/files/test_intervals.tsv
FULL_SAMPLES=/beagle3/haky/users/temi/projects/alphagenome/files/test_samples.tsv

BASE_DIR=/beagle3/haky/users/temi/projects/alphagenome/predictions/run
H5_DIR=$BASE_DIR/h5
PROFILE_DIR=$BASE_DIR/profiles
LOG_DIR=$BASE_DIR/logs

N_WORKERS=1
PAD_BINS=0
METHOD=mean
OUTPUT_TYPES="CHIP_TF"
PROFILE="--profile"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-gpus)    N_GPUS=$2;       shift 2 ;;
        --samples)   FULL_SAMPLES=$2; shift 2 ;;
        --bed)       BED=$2;          shift 2 ;;
        --outdir)
            BASE_DIR=$2
            H5_DIR=$BASE_DIR/h5
            PROFILE_DIR=$BASE_DIR/profiles
            LOG_DIR=$BASE_DIR/logs
            shift 2 ;;
        --no-profile) PROFILE="";    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_SAMPLES=$(wc -l < "$FULL_SAMPLES")

# Cap N_GPUS at N_SAMPLES — no point in empty jobs
if (( N_GPUS > N_SAMPLES )); then
    echo "Warning: N_GPUS ($N_GPUS) > N_SAMPLES ($N_SAMPLES), capping to $N_SAMPLES"
    N_GPUS=$N_SAMPLES
fi

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR"

# ── Split samples into per-GPU chunk files ────────────────────────────────────
# Stored under BASE_DIR so they persist until jobs actually run.
CHUNKS_DIR=$BASE_DIR/chunks_$(date +%Y%m%d_%H%M%S)
mkdir -p "$CHUNKS_DIR"

split -n "l/$N_GPUS" "$FULL_SAMPLES" "$CHUNKS_DIR/chunk_raw_"

i=0
for f in "$CHUNKS_DIR"/chunk_raw_*; do
    mv "$f" "$CHUNKS_DIR/chunk_${i}.txt"
    n=$(wc -l < "$CHUNKS_DIR/chunk_${i}.txt")
    echo "Chunk ${i} : ${n} samples"
    i=$(( i + 1 ))
done

# ── Submit array job ──────────────────────────────────────────────────────────
ARRAY_RANGE="0-$((N_GPUS - 1))"

EXPORT_VARS="SCRIPT=$SCRIPT"
EXPORT_VARS+=",BED=$BED"
EXPORT_VARS+=",H5_DIR=$H5_DIR"
EXPORT_VARS+=",PROFILE_DIR=$PROFILE_DIR"
EXPORT_VARS+=",LOG_DIR=$LOG_DIR"
EXPORT_VARS+=",CHUNKS_DIR=$CHUNKS_DIR"
EXPORT_VARS+=",N_WORKERS=$N_WORKERS"
EXPORT_VARS+=",PAD_BINS=$PAD_BINS"
EXPORT_VARS+=",METHOD=$METHOD"
EXPORT_VARS+=",OUTPUT_TYPES=$OUTPUT_TYPES"
EXPORT_VARS+=",PROFILE=$PROFILE"

JOB_ID=$(sbatch \
    --array="$ARRAY_RANGE" \
    --export="$EXPORT_VARS" \
    -o "$LOG_DIR/slurm_%A_%a.out" \
    -e "$LOG_DIR/slurm_%A_%a.err" \
    /beagle3/haky/users/temi/projects/alphagenome/src/predict_gpu.sbatch \
    | awk '{print $NF}')

echo ""
echo "=============================="
echo "Submitted job array : $JOB_ID"
echo "Array range         : $ARRAY_RANGE  ($N_GPUS tasks)"
echo "Samples             : $N_SAMPLES"
echo "Chunks dir          : $CHUNKS_DIR"
echo "Logs                : $LOG_DIR/slurm_${JOB_ID}_*.out"
echo "=============================="
echo ""
echo "Monitor:  squeue -j $JOB_ID"
echo "Progress: ls $H5_DIR/*.h5 2>/dev/null | wc -l  (of $N_SAMPLES)"
