#%%
import os 
from datetime import datetime
from omegaconf import DictConfig
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader

import lightning.pytorch as pl 
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


from configs import Config, DataModuleConfig, ModuleConfig
from transforms import Transform

class HFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_configs: DictConfig,
        loader_configs: DictConfig,
        transform_configs: DictConfig,
        sampling_rate: int
        ):

        super().__init__()

        self.dataset_name = dataset_configs.dataset_name
        self.data_dir = dataset_configs.dataset_dir
        self.num_classes = dataset_configs.num_classes
        self.test_size = dataset_configs.test_size
        self.train_split = dataset_configs.train_split
        self.test_spit = dataset_configs.test_split
        self.num_workers = dataset_configs.num_workers
        self.columns = dataset_configs.columns
        self.sampling_rate = sampling_rate

        self.train_transform = Transform(
            mel_params=transform_configs.mel_params,
            spectrogram_params=transform_configs.spectrogram_params,
            window_params=transform_configs.window_params,
            target_length=transform_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std
        )

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test


    def prepare_data(self):
        #pl.seed_everything(self.seed) ## needed? 
        rank_zero_info(">> Preparing data")
        if not os.path.exists(self.data_dir):
            rank_zero_info(f"[{str(datetime.now())}] Data directory {self.data_dir} does not exist. Creating it.")
            os.makedirs(self.data_dir)
        
        cache_dir_is_empty = len(os.listdir(self.data_dir)) == 0
        
        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            load_dataset(self.dataset_name, cache_dir=self.data_dir, load_from_cache_file=True)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache {self.data_dir} exists. Loading from cache in setup."
            )
    
    def setup(self, stage:str) -> None: 
        if stage == "fit" or stage is None: 
            dataset = load_dataset(
                self.dataset_name, split=self.train_split, cache_dir=self.data_dir
            )

            if self.test_size:
                split = dataset.train_test_split(
                    self.test_size,
                    shuffle=True,
                    seed=42
                )
                self.train_data = split["train"]
                self.val_data = split["test"]

            else: 
                self.train_data = dataset
                self.val_data = None

            self.train_data.set_format("numpy", columns=self.columns, output_all_columns=False)
            self.train_data = self.train_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True,decode=True))
            self.train_data.set_transform(self.train_transform)

            if self.val_data:
                self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                self.val_data = self.val_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                self.val_data.set_transform(self.train_transform)
        
        if stage == "test": 
            self.train_data = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle
        )

    





        #cache_dir_is_empty = le
        
