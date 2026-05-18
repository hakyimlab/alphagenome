#!/bin/bash
# Submit one SLURM job per GPU chunk.
#
# Usage:
#   bash launch.sh --config /path/to/run.yaml [--n-gpus N] [--samples file]
#
# The YAML config controls all inference parameters. --samples and --n-gpus
# can override the YAML samples_file and parallelism at launch time.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT=/beagle3/haky/users/temi/projects/alphagenome/src/predict.py
CONFIG=/beagle3/haky/users/temi/projects/alphagenome/configs/run.yaml
N_GPUS=8
FULL_SAMPLES=""   # if empty, read from YAML

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)    CONFIG=$2;       shift 2 ;;
        --n-gpus)    N_GPUS=$2;       shift 2 ;;
        --samples)   FULL_SAMPLES=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Read samples_file from YAML if not overridden on CLI
if [[ -z "$FULL_SAMPLES" ]]; then
    FULL_SAMPLES=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c['samples_file'])")
fi

# Read output dirs from YAML
BASE_H5=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c['output_dir'])")
PROFILE_DIR=$(python3 -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('profile_dir', c['output_dir']))")
BASE_DIR=$(dirname "$BASE_H5")
LOG_DIR=$BASE_DIR/logs
H5_DIR=$BASE_H5

N_SAMPLES=$(wc -l < "$FULL_SAMPLES")

# Cap N_GPUS at N_SAMPLES — no point in empty jobs
if (( N_GPUS > N_SAMPLES )); then
    echo "Warning: N_GPUS ($N_GPUS) > N_SAMPLES ($N_SAMPLES), capping to $N_SAMPLES"
    N_GPUS=$N_SAMPLES
fi

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR"

# ── Split samples into per-GPU chunk files ────────────────────────────────────
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
EXPORT_VARS+=",CONFIG=$CONFIG"
EXPORT_VARS+=",H5_DIR=$H5_DIR"
EXPORT_VARS+=",PROFILE_DIR=$PROFILE_DIR"
EXPORT_VARS+=",LOG_DIR=$LOG_DIR"
EXPORT_VARS+=",CHUNKS_DIR=$CHUNKS_DIR"

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
echo "Config              : $CONFIG"
echo "Array range         : $ARRAY_RANGE  ($N_GPUS tasks)"
echo "Samples             : $N_SAMPLES"
echo "Chunks dir          : $CHUNKS_DIR"
echo "Logs                : $LOG_DIR/slurm_${JOB_ID}_*.out"
echo "=============================="
echo ""
echo "Monitor:  squeue -j $JOB_ID"
echo "Progress: ls $H5_DIR/*.h5 2>/dev/null | wc -l  (of $N_SAMPLES)"
