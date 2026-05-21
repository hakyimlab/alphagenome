#!/bin/bash

#SBATCH --job-name=get_personalized_genomes
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --error=/beagle3/haky/users/temi/projects/alphagenome/logs/get_personalized_genomes.err
#SBATCH --output=/beagle3/haky/users/temi/projects/alphagenome/logs/get_personalized_genomes.out
#SBATCH --account=pi-haky

date

source ~/.bashrc
conda activate /beagle3/haky/users/temi/software/conda_envs/alphagenome-env

python3 /beagle3/haky/users/temi/projects/alphagenome/src/get_personalized_genomes.py

echo "INFO - Done!"