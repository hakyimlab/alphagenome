#!/bin/bash
# AlphaGenome unified launcher — personalized and reference modes.
#
# Personalized (default):
#   Splits a samples file across N GPUs; each GPU runs all intervals for its chunk.
#   bash alphagenomeInfer.sh --config run.yaml [--n-gpus N] [--samples file]
#
# Reference:
#   Splits intervals into N shards; each GPU processes one shard.
#   bash alphagenomeInfer.sh --config run_reference.yaml --reference [--n-gpus N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREDICT_SCRIPT="$SCRIPT_DIR/predict.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG=""
N_GPUS=8
FULL_SAMPLES=""   # personalized only; if empty, read from YAML
REFERENCE=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)     CONFIG=$2;        shift 2 ;;
        --n-gpus)     N_GPUS=$2;        shift 2 ;;
        --samples)    FULL_SAMPLES=$2;  shift 2 ;;
        --reference)  REFERENCE=true;   shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: $0 --config <run.yaml> [--reference] [--n-gpus N] [--samples file]"
    exit 1
fi

# ── Read shared paths from YAML ───────────────────────────────────────────────
_yaml() { python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"; }

BASE_H5=$(    _yaml "c['output_dir']")
PROFILE_DIR=$(_yaml "c.get('profile_dir', c['output_dir'])")
BASE_DIR=$(dirname "$BASE_H5")
LOG_DIR=$BASE_DIR/logs

mkdir -p "$BASE_H5" "$PROFILE_DIR" "$LOG_DIR"

ARRAY_RANGE="0-$((N_GPUS - 1))"

# ── Reference mode ────────────────────────────────────────────────────────────
if $REFERENCE; then
    BED_FILE=$(_yaml "c['bed_file']")
    N_INTERVALS=$(python3 -c "
import pandas as pd
df = pd.read_csv('$BED_FILE', sep='\t', header=None)
print(len(df))
")

    echo "=============================="
    echo "Mode        : reference"
    echo "Config      : $CONFIG"
    echo "Intervals   : $N_INTERVALS"
    echo "Shards      : $N_GPUS"
    echo "Output      : $BASE_H5"
    echo "Logs        : $LOG_DIR"
    echo "=============================="

    EXPORT_VARS="SCRIPT=$PREDICT_SCRIPT"
    EXPORT_VARS+=",CONFIG=$CONFIG"
    EXPORT_VARS+=",H5_DIR=$BASE_H5"
    EXPORT_VARS+=",PROFILE_DIR=$PROFILE_DIR"
    EXPORT_VARS+=",LOG_DIR=$LOG_DIR"
    EXPORT_VARS+=",N_SHARDS=$N_GPUS"

    JOB_ID=$(sbatch \
        --array="$ARRAY_RANGE" \
        --export="$EXPORT_VARS" \
        -o "$LOG_DIR/slurm_%A_%a.out" \
        -e "$LOG_DIR/slurm_%A_%a.err" \
        "$SCRIPT_DIR/predict_reference_gpu.sbatch" \
        | awk '{print $NF}')

    echo ""
    echo "Submitted   : $JOB_ID"
    echo "Array       : $ARRAY_RANGE  ($N_GPUS shards)"
    echo "Monitor     : squeue -j $JOB_ID"
    echo "Progress    : ls $BASE_H5/reference_shard*.h5 2>/dev/null | wc -l  (of $N_GPUS)"

# ── Personalized mode ─────────────────────────────────────────────────────────
else
    if [[ -z "$FULL_SAMPLES" ]]; then
        FULL_SAMPLES=$(_yaml "c['samples_file']")
    fi

    N_SAMPLES=$(wc -l < "$FULL_SAMPLES")

    if (( N_GPUS > N_SAMPLES )); then
        echo "Warning: --n-gpus ($N_GPUS) > n_samples ($N_SAMPLES) — capping to $N_SAMPLES"
        N_GPUS=$N_SAMPLES
        ARRAY_RANGE="0-$((N_GPUS - 1))"
    fi

    CHUNKS_DIR=$BASE_DIR/chunks_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$CHUNKS_DIR"

    split -n "l/$N_GPUS" "$FULL_SAMPLES" "$CHUNKS_DIR/chunk_raw_"
    i=0
    for f in "$CHUNKS_DIR"/chunk_raw_*; do
        mv "$f" "$CHUNKS_DIR/chunk_${i}.txt"
        n=$(wc -l < "$CHUNKS_DIR/chunk_${i}.txt")
        echo "Chunk ${i}: ${n} samples"
        i=$(( i + 1 ))
    done

    echo "=============================="
    echo "Mode        : personalized"
    echo "Config      : $CONFIG"
    echo "Samples     : $N_SAMPLES"
    echo "GPUs        : $N_GPUS"
    echo "Chunks dir  : $CHUNKS_DIR"
    echo "Output      : $BASE_H5"
    echo "Logs        : $LOG_DIR"
    echo "=============================="

    EXPORT_VARS="SCRIPT=$PREDICT_SCRIPT"
    EXPORT_VARS+=",CONFIG=$CONFIG"
    EXPORT_VARS+=",H5_DIR=$BASE_H5"
    EXPORT_VARS+=",PROFILE_DIR=$PROFILE_DIR"
    EXPORT_VARS+=",LOG_DIR=$LOG_DIR"
    EXPORT_VARS+=",CHUNKS_DIR=$CHUNKS_DIR"

    JOB_ID=$(sbatch \
        --array="$ARRAY_RANGE" \
        --export="$EXPORT_VARS" \
        -o "$LOG_DIR/slurm_%A_%a.out" \
        -e "$LOG_DIR/slurm_%A_%a.err" \
        "$SCRIPT_DIR/predict_gpu.sbatch" \
        | awk '{print $NF}')

    echo ""
    echo "Submitted   : $JOB_ID"
    echo "Array       : $ARRAY_RANGE  ($N_GPUS tasks)"
    echo "Monitor     : squeue -j $JOB_ID"
    echo "Progress    : ls $BASE_H5/*.h5 2>/dev/null | wc -l  (of $N_SAMPLES)"
fi
