#!/bin/bash

decays=(0.85 0.75 0.65 0.55)
# Define the learning rates you want to try
learning_rates=(1e-3 5e-4 4e-4 3e-4 2e-4 1e-4)
for decay in "${decays[@]}"; do
  # Iterate over the learning rates and run the experiment
  for lr in "${learning_rates[@]}"; do
    echo "Running experiment with lr=${lr} and decay=${decay}"
  
    # Convert the learning rate to a float for comparison
    lr_float=$(printf "%.10f" "$lr")

    python finetune.py \
      experiment=finetune_hsn.yaml \
      module.optimizer.target.lr=$lr \
      module.optimizer.extras.layer_decay=$decay
  done
done  