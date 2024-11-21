#!/usr/bin/zsh
#SBATCH --mem=200gb
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --partition=main
#SBATCH --job-name=birdMAE_pretrain_XCL_scratch_mgpu_ogg
#SBATCH --output=/mnt/work/bird2vec/logs/mgpu_ogg.log
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

SEED=$SLURM_ARRAY_TASK_ID

srun python train_ssl.py experiment=pretrain_xcl_wave.yaml

echo "Finished script."
