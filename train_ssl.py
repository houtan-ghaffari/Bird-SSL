#%%
import os 
import hydra
import lightning as L
import matplotlib.pyplot as plt  # Import matplotlib for saving spectrograms
import librosa 

from omegaconf import OmegaConf, DictConfig
import pyrootutils
from pathlib import Path 

from datamodule import HFDataModule
from transforms import Transform
from util.pylogger import get_pylogger
from build import instantiate_callbacks, build_model

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
    L.seed_everything(cfg.seed)

    log.info("Setup datamodule")
    datamodule = HFDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
        sampling_rate=cfg.module.network.sampling_rate
    )

    log.info("Setup logger")
    logger = None

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      
    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Setup model")
    model = build_model(cfg.module)

    log.info("Start training")
    trainer.fit(model=model, datamodule=datamodule)

    print("halloo")



    # datamodule.prepare_data()
    # datamodule.setup("fit")

if __name__ == "__main__":
    train()