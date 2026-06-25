#!/bin/bash
#SBATCH --job-name=alphagenome_ref
#SBATCH --account=beagle3-exusers
#SBATCH --partition=beagle3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# --output and --error are set by alphagenomeInfer.sh via -o / -e flags

set -euo pipefail

# Variables injected by alphagenomeInfer.sh via --export:
#   SCRIPT, CONFIG, H5_DIR, PROFILE_DIR, LOG_DIR, N_SHARDS

echo "=============================="
echo "Job         : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node        : $SLURMD_NODENAME"
echo "Mode        : reference"
echo "Shard       : ${SLURM_ARRAY_TASK_ID} / ${N_SHARDS}"
echo "Start       : $(date)"
echo "=============================="

module load cudnn/11.2
source /home/temi/miniconda3/etc/profile.d/conda.sh
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env-v2

mkdir -p "$H5_DIR" "$PROFILE_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
    --config      "$CONFIG" \
    --reference \
    --shard-id    "$SLURM_ARRAY_TASK_ID" \
    --n-shards    "$N_SHARDS" \
    --output-dir  "$H5_DIR" \
    --profile-dir "$PROFILE_DIR"

echo "=============================="
echo "End         : $(date)"
N_H5=$(ls "$H5_DIR"/reference_shard*.h5 2>/dev/null | wc -l)
echo "Shards done : $N_H5 / $N_SHARDS"
echo "=============================="
