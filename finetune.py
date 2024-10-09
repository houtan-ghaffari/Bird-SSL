#%%
import os 
import hydra 
import torch
import lightning as L

from omegaconf import OmegaConf, DictConfig
import pyrootutils
from pathlib import Path 

from datamodule import HFDataModule
from util.pylogger import get_pylogger
from build import instantiate_callbacks, build_model
from util.state_mapping import map_amae_checkpoint
from timm.models.vision_transformer import PatchEmbed
import torch.nn as nn

from util.pos_embed import get_2d_sincos_pos_embed_flexible

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
    "config_name": "finetune.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def finetune(cfg: DictConfig):
    log.info("Seed everything with cfg.")
    L.seed_everything(cfg.seed)

    datamodule = HFDataModule(
        dataset_configs=cfg.data.dataset,
        loader_configs=cfg.data.loaders,
        transform_configs=cfg.data.transform,
        sampling_rate=cfg.module.network.sampling_rate
    )

    log.info("Setup logger")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      
    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Setup model")
    model = build_model(cfg.module)

    pretrained_weights_path = cfg.module.network.pretrained_weights_path

    if pretrained_weights_path:
        log.info(f"Load pretrained weights from {pretrained_weights_path}")
        model.load_pretrained_weights(pretrained_weights_path)

    log.info("Start training")
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__":
    finetune()


    

