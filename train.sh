#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --partition=proxima
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=korbowody_literki
#SBATCH --output=logs/%x_%A.out
source /mnt/storage_3/home/anatolk/pl0404-03/project_data/anatolk/sprytna-pogoda/.venv/bin/activate
python src/yolo.py