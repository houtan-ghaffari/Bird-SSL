#%%
import os 
import hydra
import lightning.pytorch as pl 
import matplotlib.pyplot as plt  # Import matplotlib for saving spectrograms

from omegaconf import OmegaConf, DictConfig
import pyrootutils
from pathlib import Path 

from datamodule import HFDataModule
from transforms import Transform
from util.pylogger import get_pylogger

log = get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "train.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def train(cfg: DictConfig):
    log.info("Using config: %s", OmegaConf.to_yaml(cfg))
    log.info(f"Dataset directory:  <{os.path.abspath(cfg.paths.dataset_dir)}>")
    log.info(f"Log directory:  <{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Root directory:  <{os.path.abspath(cfg.paths.root_dir)}>")
    log.info(f"Work directory:  <{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output directory:  <{os.path.abspath(cfg.paths.output_dir)}>")

    log.info("Seed everything with cfg.")
    pl.seed_everything(cfg.seed)

    log.info("Setup datamodule")
    datamodule = HFDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
        sampling_rate=cfg.module.network.sampling_rate
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    # Save each tensor of the first batch as a spectrogram PNG
    batch = next(iter(datamodule.train_dataloader()))
    for i, audio_tensor in enumerate(batch["audio"]):
        # Convert the tensor to a numpy array and transpose it for visualization
        audio_np = audio_tensor.squeeze().numpy()  # Remove the channel dimension
        plt.figure(figsize=(10, 4))
        plt.imshow(audio_np, aspect='auto', origin='lower', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram {i}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        # Save the figure
        plt.savefig(f'spectrogram_{i}.png')
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    train()