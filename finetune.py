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
    logger = None

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      
    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Setup model")
    model = build_model(cfg.module)
    state_dict = model.state_dict()

    checkpoint = torch.load(os.path.join(cfg.paths.root_dir,"weights/amae_as2m_pretrained.pth"))
    checkpoint_model = checkpoint["model"]

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    model.load_state_dict(checkpoint_model)
    




if __name__ == "__main__":
    finetune()


    

