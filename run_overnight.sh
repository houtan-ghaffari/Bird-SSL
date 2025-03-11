#!/bin/bash

# decays=(0.85 0.75 0.65 0.55)
# # Define the learning rates you want to try
# learning_rates=(1e-3 5e-4 4e-4 3e-4 2e-4 1e-4)
# for decay in "${decays[@]}"; do
#   # Iterate over the learning rates and run the experiment
#   for lr in "${learning_rates[@]}"; do
#     echo "Running experiment with lr=${lr} and decay=${decay}"
  
#     # Convert the learning rate to a float for comparison
#     lr_float=$(printf "%.10f" "$lr")

#     python finetune.py \
#       experiment=finetune_hsn.yaml \
#       module.optimizer.target.lr=$lr \
#       module.optimizer.extras.layer_decay=$decay
#   done
# done  


#epochs=(04 09 14 19 24 29 34 39 44 49 54)
# epochs=(00 01 02 03 04)
# for ep in "${epochs[@]}"; do
#   echo "Running experiment with ep=${ep}"
  
#   sed -i "s|pretrained_weights_path:.*|pretrained_weights_path: /home/lrauch/mnt_check/finetune_xcl/runs/XCL/VIT/2025-01-06_103326/callback_checkpoints/VIT_XCL_epoch=${ep}.ckpt|g" configs/module/network/vit_large_16.yaml
  
#   python finetune.py trainer.max_epochs=30 experiment=finetune_hsn.yaml +decoder_name=standard
# done

# weight_decays=(3e-4)
# # Define the learning rates you want to try
# learning_rates=(4e-4 3e-4 2e-4 1e-4 5e-4 1e-3)
# for decay in "${weight_decays[@]}"; do
#   # Iterate over the learning rates and run the experiment
#   for lr in "${learning_rates[@]}"; do
#     echo "Running experiment with lr=${lr} and decay=${decay}"
  
#     # Convert the learning rate to a float for comparison
#     lr_float=$(printf "%.10f" "$lr")

#     python finetune.py \
#       trainer.max_epochs=30 \
#       experiment=finetune_hsn.yaml \
#       module.optimizer.target.lr=$lr \
#       module.optimizer.target.weight_decay=$decay \
#       logger.experiment_name="overnight_hsn"
#   done
# done


# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet" \
#   module.optimizer.target.lr=4e-4 \
#   module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-16_091017/callback_checkpoints/AudioMAE_XCL_epoch=99.ckpt" # swin

# # Second command
# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet" \
#   module.optimizer.target.lr=1e-4 \
#   module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2025-01-16_091017/callback_checkpoints/AudioMAE_XCL_epoch=99.ckpt" # swin

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=4e-4

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.network.ppnet.prototype_lr=1e-2

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=1e-4



# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=5e-4

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=1e-4 \
#   module.network.ppnet.prototype_lr=1e-3

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=4e-4 \
#   module.network.ppnet.prototype_lr=1e-3

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=5e-4
  
# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.network.ppnet.prototype_lr=5e-3

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=1e-4 \
#   module.network.ppnet.prototype_lr=5e-3

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=4e-4 \
#   module.network.ppnet.prototype_lr=5e-3



# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.network.pretrained_weights_path="/home/lrauch/AudioMAE_XCL_epoch150_213828.ckpt" \
#   module.optimizer.extras.layer_decay=0.5



# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=5e-5
###


# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=4e-4 \
#   module.network.pretrained_weights_path="/home/lrauch/VIT_XCL_epoch01_ft_from143556.ckpt" \
#   module.network.freeze_backbone=fals

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=3e-4 \
#   module.network.pretrained_weights_path="/home/lrauch/VIT_XCL_epoch01_ft_from143556.ckpt" \
#   module.network.freeze_backbone=true

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=5e-4 \
#   module.network.pretrained_weights_path="/home/lrauch/VIT_XCL_epoch01_ft_from143556.ckpt" \
#   module.network.freeze_backbone=true

# python finetune.py \
#   experiment=finetune_hsn_ppnet.yaml \
#   logger.experiment_name="overnight_hsn_ppnet"\
#   module.optimizer.target.lr=5e-5 \
#   module.network.pretrained_weights_path="/home/lrauch/VIT_XCL_epoch01_ft_from143556.ckpt" \
#   module.network.freeze_backbone=true

#python finetune.py experiment=paper/ablations/hsn/modelsize_base_ckpt35.yaml

#python finetune.py experiment=paper/ablations/hsn/modelsize_large_ckpt35.yaml

#modelsize
#base
# python finetune.py experiment=paper/ablations/hsn/1_modelsize_base_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl/runs/XCL/AudioMAE/2024-11-11_204649/callback_checkpoints/AudioMAE_XCL_epoch\=04.ckpt" \
#     logger.run_name="modelsize_base_ckpt5_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/1_modelsize_base_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl/runs/XCL/AudioMAE/2024-11-11_204649/callback_checkpoints/AudioMAE_XCL_epoch\=19.ckpt" \
#     logger.run_name="modelsize_base_ckpt20_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/1_modelsize_base_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl/runs/XCL/AudioMAE/2024-11-11_204649/callback_checkpoints/AudioMAE_XCL_epoch\=39.ckpt" \
#     logger.run_name="modelsize_base_ckpt40${seed}_${start_time}"

# # python finetune.py experiment=paper/ablations/hsn/modelsize_base_ckpt35.yaml \
# #     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl/runs/XCL/AudioMAE/2024-11-11_204649/callback_checkpoints/AudioMAE_XCL_epoch\=99.ckpt" \
# #     logger.run_name="modelsize_base_ckpt100_${seed}_${start_time}"

# # #large
# python finetune.py experiment=paper/ablations/hsn/1_modelsize_large_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-12-22_012826/callback_checkpoints/AudioMAE_XCL_epoch\=04.ckpt" \
#     logger.run_name="modelsize_large_ckpt5_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/1_modelsize_large_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-12-22_012826/callback_checkpoints/AudioMAE_XCL_epoch\=19.ckpt" \
#     logger.run_name="modelsize_large_ckpt20_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/1_modelsize_large_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-12-22_012826/callback_checkpoints/AudioMAE_XCL_epoch\=39.ckpt" \
#     logger.run_name="modelsize_large_ckpt40_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/modelsize_large_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_large/runs/XCL/AudioMAE/2024-12-22_012826/callback_checkpoints/AudioMAE_XCL_epoch\=99.ckpt" \
#     logger.run_name="modelsize_large_ckpt100_${seed}_${start_time}"

# #huge
# python finetune.py experiment=paper/ablations/hsn/modelsize_huge_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_huge/runs/XCL/AudioMAE/2025-01-05_145126/callback_checkpoints/AudioMAE_XCL_epoch\=49.ckpt" \
#     logger.run_name="modelsize_huge_ckpt50_${seed}_${start_time}"

# python finetune.py experiment=paper/ablations/hsn/modelsize_huge_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_huge/runs/XCL/AudioMAE/2025-01-05_145126/callback_checkpoints/AudioMAE_XCL_epoch\=59.ckpt" \
#     logger.run_name="modelsize_huge_ckpt60_${seed}_${start_time}"   

# python finetune.py experiment=paper/ablations/hsn/modelsize_huge_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_huge/runs/XCL/AudioMAE/2025-01-05_145126/callback_checkpoints/AudioMAE_XCL_epoch\=79.ckpt" \
#     logger.run_name="modelsize_huge_ckpt80_${seed}_${start_time}"   

# python finetune.py experiment=paper/ablations/hsn/modelsize_huge_ckpt35.yaml \
#     module.network.pretrained_weights_path="/home/lrauch/mnt_check/pretrain_xcl_wave_huge/runs/XCL/AudioMAE/2025-01-05_145126/callback_checkpoints/AudioMAE_XCL_epoch\=99.ckpt" \
#     logger.run_name="modelsize_huge_ckpt100_${seed}_${start_time}"   

#decoder
# python finetune.py experiment=paper/ablations/hsn/3_0.3mixup_large_ckpt100.yaml
# echo "test"
# python finetune.py experiment=paper/ablations/hsn/3_0.5mixup_large_ckpt100.yaml
# python finetune.py experiment=paper/ablations/hsn/3_0.7mixup_large_ckpt100.yaml

# python finetune.py experiment=paper/ablations/hsn/2_decoder_swin_ckpt35.yaml
# python finetune.py experiment=paper/ablations/hsn/2_decoder_swinv2_ckpt35.yaml



#more than 100
# python finetune.py experiment=paper/ablations/hsn/4_epochs120_large_ckpt120.yaml
# python finetune.py experiment=paper/ablations/hsn/4_epochs140_large_ckpt140.yaml
# python finetune.py experiment=paper/ablations/hsn/4_epochs150_large_ckpt150.yaml    

# python finetune.py experiment=paper/ablations/hsn/5_batchsize256_large_ckpt100.yaml
# python finetune.py experiment=paper/ablations/hsn/6_maskratio0.65_large_ckpt60.yaml
# python finetune.py experiment=paper/ablations/hsn/6_maskratio0.7_large_ckpt60.yaml
# python finetune.py experiment=paper/ablations/hsn/6_maskratio0.75std_large_ckpt60.yaml
# python finetune.py experiment=paper/ablations/hsn/7_datasetsizeALL_large_ckpt99.yaml        




# python finetune.py experiment=paper/ablations/hsn_pp_frozen/150_epochs_mixup_pp.yaml

# python finetune.py experiment=paper/ablations/hsn_pp_frozen/amae_ppnet.yaml


python finetune.py experiment=paper/ablations/hsn_mlp/100_epochs_mixup_mlp_base.yaml
python finetune.py experiment=paper/ablations/hsn_mlp/100_epochs_mixup_mlp_huge.yaml
python finetune.py experiment=paper/ablations/hsn_mlp/amae_mlp.yaml
python finetune.py experiment=paper/ablations/hsn_linear/100_epochs_mixup_linear_huge.yaml

