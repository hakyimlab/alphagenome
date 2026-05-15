#!/bin/bash
#SBATCH --job-name=alphagenome_test
#SBATCH --account=beagle3-exusers
#SBATCH --partition=beagle3
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/test_alphagenome.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/test_alphagenome.err

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT=/beagle3/haky/users/temi/projects/alphagenome/src/predict.py
BED=/beagle3/haky/users/temi/projects/alphagenome/files/test_intervals.tsv
FULL_SAMPLES=/beagle3/haky/users/temi/projects/alphagenome/files/test_samples.tsv

BASE_DIR=/beagle3/haky/users/temi/projects/alphagenome/predictions/test_run
H5_DIR=$BASE_DIR/h5
PROFILE_DIR=$BASE_DIR/profiles
LOG_DIR=$BASE_DIR/logs

N_TEST_SAMPLES=2   # total samples to use in this test (split evenly across GPUs)
N_WORKERS=4         # Parsl I/O threads per GPU process
PAD_BINS=0
METHOD=mean
OUTPUT_TYPES="CHIP_TF"
PROFILE="--profile"   # set to "" to disable profiling

# ── Environment ───────────────────────────────────────────────────────────────
module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR" \
         /beagle3/haky/users/temi/projects/alphagenome/logs

# ── Detect available GPUs ─────────────────────────────────────────────────────
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs found  : $N_GPUS"
echo "H5 dir      : $H5_DIR"
echo "Profile dir : $PROFILE_DIR"
echo "Log dir     : $LOG_DIR"
echo "Start       : $(date)"
echo "=============================="

# ── Split samples evenly across GPUs ─────────────────────────────────────────
mkdir -p /scratch/midway2/temi
TMPDIR=$(mktemp -d /scratch/midway2/temi/alphagenome_XXXXXX)

head -"$N_TEST_SAMPLES" "$FULL_SAMPLES" > "$TMPDIR/test_samples.txt"
split -n "l/$N_GPUS" "$TMPDIR/test_samples.txt" "$TMPDIR/samples_gpu_"

i=0
for f in "$TMPDIR"/samples_gpu_*; do
    mv "$f" "$TMPDIR/samples_gpu_${i}.txt"
    echo "GPU ${i} samples : $(tr '\n' ' ' < "$TMPDIR/samples_gpu_${i}.txt")"
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
        $PROFILE \
        > "$LOG_DIR/gpu_${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=($!)
done

# ── Wait for all processes and collect exit codes ─────────────────────────────
echo ""
OVERALL=0
for (( gpu=0; gpu<${#PIDS[@]}; gpu++ )); do
    wait "${PIDS[$gpu]}"; STATUS=$?
    echo "GPU ${gpu} exit status : $STATUS"
    [[ $STATUS -ne 0 ]] && OVERALL=1
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "End         : $(date)"
echo "HDF5 files  :"
ls -lh "$H5_DIR"/*.h5      2>/dev/null || echo "  (none)"
echo "Profile files :"
ls -lh "$PROFILE_DIR"/*.tsv 2>/dev/null || echo "  (none)"
echo "GPU logs :"
ls -lh "$LOG_DIR"/*.log     2>/dev/null || echo "  (none)"
echo "=============================="

rm -rf "$TMPDIR"
exit $OVERALL
