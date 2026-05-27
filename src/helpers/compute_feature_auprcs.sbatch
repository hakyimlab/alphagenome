#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=feature_auprcs
#SBATCH --account=pi-haky
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/compute_feature_auprcs_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/compute_feature_auprcs_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=caslake

date

source ~/.bashrc
conda activate /beagle3/haky/users/shared_software/TFXcan-pipeline-tools

# Args: ${1}=test_data_file ${2}=output_file ${3}=ncores (optional, default 8)
CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/compute_feature_auprcs.R \
    --test_data_file \"${1}\" \
    --output_file \"${2}\""

if [[ -n "${3}" ]]; then
    CMD="$CMD --ncores ${3}"
fi
eval $CMD

status=$?
echo "Exit status of job is: $status"
