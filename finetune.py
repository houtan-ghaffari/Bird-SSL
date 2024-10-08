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

    img_size = (cfg.data.dataset.target_length, 128)
    in_chans = 1
    model.patch_embed = PatchEmbed(img_size, 16, in_chans, 768)
    num_patches = model.patch_embed.num_patches
    num_patches = 512 #audioset
    model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False)
    
    state_dict = model.state_dict()

    checkpoint = torch.load(os.path.join(cfg.paths.root_dir,"weights/amae_as2m_pretrained.pth"))
    pretrained_state_dict = checkpoint["model"]
    #pretrained_state_dict = map_amae_checkpoint(pretrained_state_dict)

    for k in ['head.weight', 'head.bias']:
        if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del pretrained_state_dict[k]

    for k in list(pretrained_state_dict.keys()):
        if 'decoder' in k:
            print(f"Removing key {k} from pretrained checkpoint")
            del pretrained_state_dict[k]

    model.load_state_dict(pretrained_state_dict, strict=False)

    log.info("Start training")
    trainer.fit(model=model, datamodule=datamodule)
    


if __name__ == "__main__":
    finetune()


    

