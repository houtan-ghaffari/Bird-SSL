#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:2
#SBATCH --mem=500gb
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_XCL_swin_large_nomix
#SBATCH --output=/mnt/work/bird2vec/logs/birdMAE_XCL_swin_large_%N_%t_nomix.log
#SBATCH --time=96:00:00
#SBATCH --nodelist=gpu-l40s-1

####SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4,gpu-a100-4

######,gpu-a100-1,gpu-a100-2
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

hostname
srun python train_ssl.py \
        experiment=pretrain_xcl_wave_large.yaml \
        trainer.devices=2 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16 \
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"

echo "Finished script."