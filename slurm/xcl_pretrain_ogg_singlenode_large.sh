#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:2
#SBATCH --mem=128gb
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_pretrain_XCL_scratch_mgpu_ogg_2large
#SBATCH --output=/mnt/work/bird2vec/logs/mgpu_ogg2large_%N_%t.log
#SBATCH --time=48:00:00
#BATCH --nodelist=gpu-a100-5

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
srun python train_ssl.py experiment=pretrain_xcl_wave_large.yaml \
        trainer.devices=2 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16

echo "Finished script."
