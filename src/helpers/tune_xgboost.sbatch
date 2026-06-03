#!/bin/bash
#SBATCH --job-name=tune_xgboost
#SBATCH --account=pi-haky
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/tune_xgboost_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/tune_xgboost_%j.err

# Usage:
#   sbatch tune_xgboost.sbatch \
#       <train_data_file> <model_rds_file> <weights_file> \
#       <tuning_results_file> <n_configs> <nfolds> <metadata> [features_group]
#
# Example:
#   sbatch tune_xgboost.sbatch \
#       /scratch/midway2/temi/data/train.tsv.gz \
#       /scratch/midway2/temi/models/best_model.rds \
#       /scratch/midway2/temi/models/weights.tsv.gz \
#       /scratch/midway2/temi/models/tuning_results.tsv \
#       50 5 ag_xgb_tune embeddings_128bp

source ~/.bashrc
conda activate /beagle3/haky/users/temi/software/conda_envs/renviron

set -euo pipefail

date
mkdir -p /beagle3/haky/users/temi/projects/alphagenome/logs

CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/tune_xgboost.R \
    --train_data_file     \"${1}\" \
    --model_rds_file      \"${2}\" \
    --weights_file        \"${3}\" \
    --tuning_results_file \"${4}\" \
    --n_configs           ${5} \
    --nfolds              ${6} \
    --metadata            ${7} \
    --ncores              ${SLURM_CPUS_PER_TASK:-12}"

if [[ -n "${8:-}" ]]; then
    CMD="$CMD --features_group \"${8}\""
fi

eval $CMD

status=$?
echo "Exit status: $status"
date