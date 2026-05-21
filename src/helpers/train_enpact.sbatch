#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G    
#SBATCH --job-name=train_enpact
#SBATCH --account=pi-haky
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/train_enpact_%j.out
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/train_enpact_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=bigmem

date

source ~/.bashrc
conda activate /beagle3/haky/users/shared_software/TFXcan-pipeline-tools

CMD="Rscript /beagle3/haky/users/temi/projects/alphagenome/src/helpers/train_enpact.R \
    --train_data_file \"${1}\" \
    --model_rds_file \"${2}\" \
    --nfolds ${3} \
    --weights_file \"${4}\" \
    --metadata ${5}"

if [[ -n "${6}" ]]; then
    CMD="$CMD --features_group \"${6}\""
fi
eval $CMD

status=$?
echo "Exit status of job is: $status"