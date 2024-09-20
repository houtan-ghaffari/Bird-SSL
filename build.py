import hydra
from omegaconf import DictConfig, OmegaConf
from models import AudioMAE
from util import pylogger
import lightning as L
log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(cfg_callbacks: DictConfig):
    callbacks = []

    if not cfg_callbacks:
        log.warning("No callbacks found")

    for _, cb_config in cfg_callbacks.items():
        if isinstance(cb_config, DictConfig) and "_target_" in cb_config:
            log.info(f"Instantiating callback <{cb_config._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_config))
    return callbacks


def build_model(cfg_module: DictConfig):
    if cfg_module.network.name == "AudioMAE":
        module = AudioMAE(
            norm_layer=cfg_module.network.norm_layer,
            norm_pix_loss=cfg_module.network.norm_pix_loss,
            cfg_encoder=cfg_module.network.encoder,
            cfg_decoder=cfg_module.network.decoder,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler
        )

    else:
        raise ValueError(f"Model {cfg_module.network.name} not found")

    return module