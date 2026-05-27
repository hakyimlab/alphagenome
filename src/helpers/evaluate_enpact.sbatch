#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=evaluate_enpact
#SBATCH --account=pi-haky
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/evaluate_enpact_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/evaluate_enpact_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=caslake

date

source ~/.bashrc
conda activate /beagle3/haky/users/shared_software/TFXcan-pipeline-tools

# Args: ${1}=test_data_file ${2}=weights_file ${3}=predictions_file ${4}=metadata ${5}=features_group (optional)
CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/evaluate_enpact.R \
    --test_data_file \"${1}\" \
    --weights_file \"${2}\" \
    --predictions_file \"${3}\" \
    --metadata ${4}"

if [[ -n "${5}" ]]; then
    CMD="$CMD --features_group \"${5}\""
fi
eval $CMD

status=$?
echo "Exit status of job is: $status"
