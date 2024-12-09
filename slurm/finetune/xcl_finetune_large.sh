#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:4
#SBATCH --mem=500gb
#SBATCH --partition=main
#SBATCH --job-name=finetune_XCL_swin_large
#SBATCH --output=/mnt/work/bird2vec/logs/finetune/finetune_XCL_swin_large_%N_%t.log
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
#####export NCCL_DEBUG=INFO
####export NCCL_DEBUG_SUBSYS=ALL
#######export CUDA_VISIBLE_DEVICES=3,2,1,0
NUM_GPUS=$SLURM_GPUS_ON_NODE

hostname
srun python train_ssl.py \
        experiment=pretrain_xcl_wave_large.yaml \
        trainer.devices=4 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16 \
        module.network.pretrained_weights_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_large_swin/runs/XCL/AudioMAE/2024-12-04_205512/callback_checkpoints/last.ckpt"
        #ckpt_path="/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-11-23_123703/callback_checkpoints/last.ckpt"

# Capture the exit status of the Python script
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Python script encountered an error (exit status: $EXIT_STATUS). Waiting for 60 seconds before exiting..."
    sleep 60m  # Adjust the wait time as needed
else
    echo "Python script completed successfully."
fi

echo "Finished script."
