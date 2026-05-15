#!/bin/bash
# Run the AlphaGenome pipeline on whichever GPUs are visible.
# Intended for interactive use inside screen/tmux on an allocated node.
#
# Usage:
#   bash predict_run.sh [--samples path] [--bed path] [--outdir path] [--no-profile]
#
# Defaults are the same as predict_run.sbatch.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
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
        --samples)   FULL_SAMPLES=$2; shift 2 ;;
        --bed)       BED=$2;          shift 2 ;;
        --outdir)    BASE_DIR=$2; H5_DIR=$BASE_DIR/h5; PROFILE_DIR=$BASE_DIR/profiles; LOG_DIR=$BASE_DIR/logs; shift 2 ;;
        --no-profile) PROFILE="";    shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR" \
         /beagle3/haky/users/temi/projects/alphagenome/logs

# ── Detect GPUs ───────────────────────────────────────────────────────────────
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_SAMPLES=$(wc -l < "$FULL_SAMPLES")
RUN_ID="$$"   # use PID as run identifier in place of SLURM_JOB_ID

echo "=============================="
echo "PID         : $RUN_ID"
echo "Node        : $(hostname)"
echo "GPUs found  : $N_GPUS"
echo "Samples     : $N_SAMPLES"
echo "H5 dir      : $H5_DIR"
echo "Profile dir : $PROFILE_DIR"
echo "Log dir     : $LOG_DIR"
echo "Start       : $(date)"
echo "=============================="

# ── Split samples across GPUs ─────────────────────────────────────────────────
mkdir -p /scratch/midway2/temi
TMPDIR=$(mktemp -d /scratch/midway2/temi/alphagenome_XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

split -n "l/$N_GPUS" "$FULL_SAMPLES" "$TMPDIR/samples_gpu_"

i=0
for f in "$TMPDIR"/samples_gpu_*; do
    mv "$f" "$TMPDIR/samples_gpu_${i}.txt"
    n=$(wc -l < "$TMPDIR/samples_gpu_${i}.txt")
    echo "GPU ${i} : ${n} samples"
    i=$(( i + 1 ))
done

# ── Launch one predict.py per GPU ─────────────────────────────────────────────
echo ""
echo "Launching..."
PIDS=()

for (( gpu=0; gpu<N_GPUS; gpu++ )); do
    SAMPLE_FILE="$TMPDIR/samples_gpu_${gpu}.txt"
    [[ -f "$SAMPLE_FILE" ]] || { echo "No sample file for GPU $gpu, skipping"; continue; }

    echo "Launching GPU ${gpu} ($(wc -l < "$SAMPLE_FILE") samples) ..."
    CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
        --samples-file "$SAMPLE_FILE" \
        --bed-file     "$BED" \
        --output-dir   "$H5_DIR" \
        --profile-dir  "$PROFILE_DIR" \
        --device       cuda:0 \
        --n-workers    "$N_WORKERS" \
        --pad-bins     "$PAD_BINS" \
        --method       "$METHOD" \
        --output-types $OUTPUT_TYPES \
        --skip-existing \
        $PROFILE \
        > "$LOG_DIR/gpu_${gpu}_${RUN_ID}.log" 2>&1 &
    PIDS+=($!)
done

# ── Wait and collect exit codes ───────────────────────────────────────────────
echo ""
OVERALL=0
for (( gpu=0; gpu<${#PIDS[@]}; gpu++ )); do
    wait "${PIDS[$gpu]}"; STATUS=$?
    echo "GPU ${gpu} exit status : $STATUS"
    [[ $STATUS -ne 0 ]] && OVERALL=1
done

# ── Summary ───────────────────────────────────────────────────────────────────
N_H5=$(ls "$H5_DIR"/*.h5 2>/dev/null | wc -l)

echo ""
echo "=============================="
echo "End              : $(date)"
echo "HDF5 completed   : $N_H5 / $N_SAMPLES"
echo "Failed samples   :"
cat "$H5_DIR/failed_samples.txt" 2>/dev/null || echo "  (none)"
echo "=============================="

exit $OVERALL
