#!/bin/bash

# Define the learning rates you want to try
learning_rates=(5e-3 3e-3 1e-3 5e-4 3e-4 1e-4 5e-5 3e-5 1e-5)

# Iterate over the learning rates and run the experiment
for lr in "${learning_rates[@]}"; do
  echo "Running experiment with lr=${lr}"
  python finetune.py experiment=finetune_hsn.yaml module.optimizer.target.lr=$lr
done
