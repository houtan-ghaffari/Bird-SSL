#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:4
#SBATCH --mem=128gb
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_pretrain_XCL_scratch_mgpu_birdset
#SBATCH --output=/mnt/work/bird2vec/logs/mgpu_birdset_spec_%N_%t.log
#SBATCH --time=60:00:00
###SBATCH --exclude=gpu-v100-3
####SBATCH --nodelist=gpu-a100-5

#SBATCH --exclude=gpu-v100-1,gpu-v100-2,gpu-v100-3,gpu-v100-4
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
        trainer.devices=4 \
        +trainer.num_nodes=1 \
        trainer.precision=bf16

echo "Finished script."
