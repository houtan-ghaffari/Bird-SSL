#%%
import os 

from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Union

this_file = Path(__file__)
project_root = this_file.parent

@dataclass
class Config: 
    cache_dir: str = os.path.join(project_root, "data/HSN") # correct this later
    log_dir: str = os.path.join(project_root, "logs")
    ckpt_dir: str = os.path.join(project_root, "checkpoints")
    seed: int = 42

@dataclass
class ModuleConfig: 
    model_name: str = "AudioMAE"
    learning_rate: float = 1e-4
    sampling_rate: int = 32_000

@dataclass
class DataModuleConfig: 
    dataset_name: str = "ashraq/esc50"
    num_classes: int = 50
    batch_size: int = 32
    train_split: str = "train"
    test_split: str = "test"
    train_size: float = 0.8
    num_workers: int = cpu_count() // 2

    cache_dir: str = field(init=False)

    def __post_init__(self):
        dataset_suffix = self.dataset_name.split('/')[-1]
        self.cache_dir = os.path.join(Config().cache_dir, dataset_suffix)

@dataclass 
class TrainerConfig: 
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    strategy: str = "auto"
    precision: str = "16-mixed"
    max_epochs: int = 1 

# %%
