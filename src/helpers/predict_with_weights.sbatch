#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=predict_weights
#SBATCH --account=pi-haky
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/predict_weights_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/predict_weights_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=caslake

# Usage:
#   sbatch predict_with_weights.sbatch \
#       /path/to/ag_array_ifg.rds \
#       /path/to/weights_dir/ \
#       /path/to/output_dir/

date

source ~/.bashrc
conda activate /beagle3/haky/users/shared_software/TFXcan-pipeline-tools

Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/predict_with_weights.R \
    --array_rds   "${1}" \
    --weights_dir "${2}" \
    --output_dir  "${3}"

status=$?
echo "Exit status: $status"
date