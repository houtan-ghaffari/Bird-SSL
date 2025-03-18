#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH --partition=main
#SBATCH --job-name=ppnet_fewshot_%A_%a
#SBATCH --output=/mnt/work/bird2vec/logs/fewshot/ppnet_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --array=0-62%3
#SBATCH --nodelist=gpu-l40s-1

date;hostname;pwd
source /mnt/home/lrauch/.zshrc
echo Activate conda
conda activate gadme_v1
echo $PYTHONPATH

cd /mnt/home/lrauch/projects/birdMAE/

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# Define arrays for datasets and seeds
DATASETS=("hsn" "nbp" "nes" "per" "pow" "sne" "ssw" "uhh")
SHOTS=("1shot" "5shot" "10shot")
SEEDS=(1 2 3)

# Calculate indices
ARRAY_INDEX=$SLURM_ARRAY_TASK_ID
DATASET_INDEX=$((ARRAY_INDEX / 9))  # 9 = 3 shots * 3 seeds
SHOT_INDEX=$((ARRAY_INDEX % 9 / 3))  # 3 seeds
SEED_INDEX=$((ARRAY_INDEX % 3))

# Get the actual values
DATASET=${DATASETS[$DATASET_INDEX]}
SHOT=${SHOTS[$SHOT_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# Construct the config path
CONFIG_PATH="experiment=paper/fewshot/ppnet/${DATASET}${SHOT}.yaml"

hostname
srun python finetune.py \
        $CONFIG_PATH \
        module.network.pretrained_weights_path="'/mnt/work/bird2vec/logs_pretrain_audioset_MAE/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-13_213828/callback_checkpoints/AudioMAE_XCL_epoch=149.ckpt'" \
        seed=$SEED

# Print information about the job
echo "Running configuration:"
echo "Dataset: $DATASET"
echo "Shot: $SHOT"
echo "Seed: $SEED"
echo "Config path: $CONFIG_PATH" 