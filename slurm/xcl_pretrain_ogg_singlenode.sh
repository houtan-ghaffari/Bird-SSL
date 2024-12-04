#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:8
#SBATCH --mem=600gb
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_XCL_swin_base
#SBATCH --output=/mnt/work/bird2vec/logs/birdMAE_XCL_swin_base_%N_%t_res.log
#SBATCH --time=96:00:00
###SBATCH --exclude=gpu-v100-3
#SBATCH --nodelist=gpu-l40s-1

###SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4
######,gpu-a100-1,gpu-a100-2
#####SBATCH --nodelist=gpu-a100-5
####SBATCH --array=3-3%3

date;hostname;pwd
source /mnt/home/lrauch/.zshrc
#source ~/envs/gadme_v1/bin/activate
echo Activate conda
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/lrauch/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

NUM_GPUS=$SLURM_GPUS_ON_NODE

hostname
srun python train_ssl.py experiment=pretrain_xcl_wave.yaml \
        trainer.devices=2 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16 \
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave/runs/XCL/AudioMAE/2024-11-29_132014/callback_checkpoints/last.ckpt"


echo "Finished script."
