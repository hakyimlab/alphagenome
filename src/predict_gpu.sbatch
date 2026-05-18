#!/bin/bash
#SBATCH --job-name=alphagenome
#SBATCH --account=beagle3-exusers
#SBATCH --partition=beagle3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# --output and --error are set by launch.sh via -o / -e flags

set -euo pipefail

# All variables injected by launch.sh via --export:
#   SCRIPT, CONFIG, H5_DIR, PROFILE_DIR, LOG_DIR, CHUNKS_DIR

SAMPLES_FILE="$CHUNKS_DIR/chunk_${SLURM_ARRAY_TASK_ID}.txt"

if [[ ! -f "$SAMPLES_FILE" ]]; then
    echo "No sample file for task $SLURM_ARRAY_TASK_ID — skipping"
    exit 0
fi

N_SAMPLES=$(wc -l < "$SAMPLES_FILE")

echo "=============================="
echo "Job         : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node        : $SLURMD_NODENAME"
echo "Samples     : $N_SAMPLES"
echo "Chunk file  : $SAMPLES_FILE"
echo "Start       : $(date)"
echo "=============================="

module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --config       "$CONFIG" \
    --samples-file "$SAMPLES_FILE" \
    --output-dir   "$H5_DIR" \
    --profile-dir  "$PROFILE_DIR"

echo "=============================="
echo "End         : $(date)"
N_H5=$(ls "$H5_DIR"/*.h5 2>/dev/null | wc -l)
echo "HDF5 so far : $N_H5"
echo "=============================="
