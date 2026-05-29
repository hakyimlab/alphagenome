#!/bin/bash
#SBATCH --job-name=train_xgboost
#SBATCH --account=pi-haky
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/train_xgboost_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/train_xgboost_%j.err

# Usage (positional arguments match train_enpact.sbatch):
#   sbatch train_xgboost.sbatch \
#       <train_data_file> <model_rds_file> <nfolds> <weights_file> <metadata> [features_group]

source ~/.bashrc
conda activate /beagle3/haky/users/temi/software/conda_envs/renviron

set -euo pipefail

date
mkdir -p /beagle3/haky/users/temi/projects/alphagenome/logs

CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/train_xgboost.R \
    --train_data_file \"${1}\" \
    --model_rds_file  \"${2}\" \
    --nfolds          ${3} \
    --weights_file    \"${4}\" \
    --metadata        ${5} \
    --ncores          ${SLURM_CPUS_PER_TASK:-12}"

if [[ -n "${6:-}" ]]; then
    CMD="$CMD --features_group \"${6}\""
fi

eval $CMD

status=$?
echo "Exit status: $status"
date
