# `BirdMAE`: A Bird Bioacoustic Foundation Model 

This repository hosts the code of "**Can Masked Autoencoders listen to Bird?**".  Masked Autoencoders (MAEs) pretrained on AudioSet fail to capture the fine-grained acoustic characteristics of specialized domains such as bioacoustic monitoring. Bird sound classification is critical for assessing environmental health, yet general-purpose models inadequately address its unique acoustic challenges. To address this, we introduce Bird-MAE, a domain-specialized MAE pretrained on the large-scale BirdSet dataset. We explore adjustments to pretraining, fine-tuning and utilizing frozen representations. Bird-MAE achieves state-of-the-art results across all BirdSet downstream tasks, substantially improving multi-label classification performance compared to the general-purpose Audio-MAE baseline. 

<br>
<div align="center">
  <img src="https://github.com/DBD-research-group/Bird-MAE/blob/main/docs/imgs/GA.png" alt="logo", width=700>
</div>
<br>

## Installation
You can install the environment to reproduce the results with: 

```
conda create -n birdmae python=3.10.14
pip  install -r requirements.txt
```

## Data Preparation
The paper includes three different types of data that is used: 

- `Pretraining data` for training the Bird-MAE model
- `Downstream data` for doing the fine-tuning/probing on the complete downstream tasks from BirdSet
- `Few-shot data` for doing the probing on few-shot setting in BirdSet

For each data type, you have to download and prepare the data. Of course, you could also download the checkpoints of the respective model and skip the pretraining. The downloading and preparation is available in the `util/prepare_data.py` file. 
Be sure to change the respective paths in the scripts. Example:

```python
example
```

## Pretraining on `BirdSet`

The main pretraining script is `pretrain.py`. The experiments are managed by hydra. The pretraining configs for the base, large and huge model are available in `configs/experiment/paper/pretrain`. The respective slurm files can be found in `slurm/pretrain/{base,large,huge}`. You have to change the paths etc. Example: 
```
sbatch slurm/pretrain/large/large/large.sh
```

## Multi-Label Benchmark on `BirdSet`

All config files of the experiments for the multi-label benchmark (with fine-tuning and linear probing) are available in `configs/experiment/bigshot`. Example: 

``` bash
python train.py experiment="paper/bigshot/$model/$type/$head/$dataset"
```

## Multi-Label Few-Shot Benchmark on `BirdSet`

All config files of the experiments for the multi-label benchmark (with fine-tuning and linear probing) are available in `configs/experiment/fewshot`. Example: 

``` bash
python train.py experiment="paper/fewshot/$probing/$probing/$dataset_kshots"
```

## Checkpoints
- Bird-MAE Base: Link
- Bird-MAE Large: Link
- Bird-MAE Huge: Link

## Citation 
```
@article{rauch2025birdmae,
      title={Can Masked Autoencoders Also Listen to Birds?}, 
      author={Lukas Rauch and Ilyass Moummad and René Heinrich and Alexis Joly and Bernhard Sick and Christoph Scholz},
      year={2025},
      journal={arXiv:2504.12880},
}
```

