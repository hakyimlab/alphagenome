#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=gather_predictions
#SBATCH --account=pi-haky
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/gather_predictions_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/gather_predictions_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=caslake

# Usage:
#   sbatch gather_predictions.sbatch \
#       /path/to/h5_dir/ \
#       /path/to/output.rds.gz \
#       [n_cores]          # optional, defaults to cpus-per-task - 2

date

source ~/.bashrc
conda activate /beagle3/haky/users/temi/software/conda_envs/renviron

N_CORES="${3:-10}"

Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/gather_predictions.R \
    --h5_dir   "${1}" \
    --output   "${2}" \
    --n_cores  "${N_CORES}"

status=$?
echo "Exit status: $status"
date