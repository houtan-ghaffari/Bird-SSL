#!/usr/bin/zsh
#SBATCH --job-name=prepare_birdset
#SBATCH --output=/mnt/work/bird2vec/logs_mw/prepare_birdset_%N.log
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --nodelist=gpu-v100-4
#SBATCH --gres=gpu:0

date;hostname;pwd
source /mnt/stud/home/mwirth/.zshrc
conda activate GADME

cd /mnt/stud/home/mwirth/projects/birdMAE/

srun python notebooks/prepare_birdset.py
