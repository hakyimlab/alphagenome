#!/bin/bash
#SBATCH --job-name=evaluate_xgboost
#SBATCH --account=pi-haky
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/evaluate_xgboost_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/evaluate_xgboost_%j.err

# Usage (positional arguments):
#   sbatch evaluate_xgboost.sbatch \
#       <test_data_file> <model_rds_file> <predictions_file> <metadata> [features_group]

source ~/.bashrc
conda activate /beagle3/haky/users/temi/software/conda_envs/renviron

set -euo pipefail

date
mkdir -p /beagle3/haky/users/temi/projects/alphagenome/logs

CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/evaluate_xgboost.R \
    --test_data_file   \"${1}\" \
    --model_rds_file   \"${2}\" \
    --predictions_file \"${3}\" \
    --metadata         ${4}"

if [[ -n "${5:-}" ]]; then
    CMD="$CMD --features_group \"${5}\""
fi

eval $CMD

status=$?
echo "Exit status: $status"
date
