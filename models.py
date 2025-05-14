import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import datasets
from torchmetrics import MetricCollection
from timm.models.vision_transformer import PatchEmbed
from timm.models.vision_transformer import Block
#from timm.models.swin_transformer import SwinTransformerBlock
from util.swin_transformer import SwinTransformerBlock
from util.swin_transformerv2 import SwinTransformerV2Block
#from timm.models.swin_transformer import SwinTransformerBlock

from timm.models.vision_transformer import VisionTransformer
from timm.optim.optim_factory import param_groups_weight_decay
from timm.models.layers import trunc_normal_
from functools import partial
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible
from util.patch_embed import PatchEmbed_new, PatchEmbed_org
from util.lamb import LAMB
from transformers import get_cosine_schedule_with_warmup
import math
from torch.optim.lr_scheduler import _LRScheduler
from util.lr_decay import param_groups_lrd
from models_ppnet import LinearLayerWithoutNegativeConnections


class MAE_Encoder(nn.Module):
    def __init__(self, 
                 img_size_x,
                 img_size_y, 
                 patch_size, 
                 in_chans, 
                 embed_dim, 
                 depth, 
                 num_heads, 
                 mlp_ratio, 
                 norm_layer, 
                 pos_trainable
                 ):
        super().__init__()
        # input: (Batch, Channel, Height 128, Width 1024)
        # output: (Batch, #Patch, Embed)
        self.patch_embed = PatchEmbed_org((img_size_x, img_size_y), patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1, 1, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #norm_layer = nn.LayerNorm()
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length(number of patches), dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1) # restore the oriignal order

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio):
        # embed patches through encoder
        x = self.patch_embed(x) # batch size, 1, width, height

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # expands on batch
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
            
class MAE_Decoder(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 decoder_embed_dim, 
                 decoder_depth, 
                 decoder_num_heads, 
                 decoder_mode,
                 mlp_ratio, 
                 norm_layer, 
                 num_patches,
                 pos_trainable,
                 patch_size,
                 in_chans,
                 no_shift,
                 target_length
        ):
        super().__init__()
        self.decoder_mode = decoder_mode
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)
        self.no_shift = no_shift
        norm_layer = partial(nn.LayerNorm, eps=1e-6) # for both?
    
        if "swin" in self.decoder_mode: #!= 0 --> should be 0 in the original implementation
            decoder_modules = []
            window_size = (4,4)
            if target_length == 1024: #10 seconds
                feat_size = (64,8)
            elif target_length == 512: #5 seconds
                feat_size = (32,8)
            else:
                raise ValueError("Target length not supported for swin decoder")

            for i in range(decoder_depth): # is 16 for swin (but has practically the same result as 8 in the paper)
                if no_shift: # shift is true
                    shift_size = (0,0)
                else:
                    if (i % 2) == 0: # every second block
                        shift_size = (0,0)
                    else:
                        shift_size = (2,0)

                if self.decoder_mode == "swin":
                    decoder_modules.append(
                        SwinTransformerBlock(
                            dim=decoder_embed_dim,
                            num_heads=decoder_num_heads,
                            feat_size=feat_size,
                            window_size=window_size,
                            shift_size=shift_size,
                            mlp_ratio=mlp_ratio,
                            drop=0.0,
                            drop_attn=0.0,
                            drop_path=0.0,
                            extra_norm=False,
                            sequential_attn=False,
                            norm_layer=norm_layer
                        )
                    )
                elif self.decoder_mode == "swinv2":
                    decoder_modules.append(
                        SwinTransformerV2Block(
                            dim=decoder_embed_dim,
                            input_resolution=feat_size,
                            num_heads=decoder_num_heads,
                            window_size=window_size,
                            shift_size=shift_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            proj_drop=0.0,
                            attn_drop=0.0,
                            drop_path=0.0,
                            act_layer=nn.GELU,
                            norm_layer=nn.LayerNorm
                        )
                    )
                else:
                    raise ValueError("Decoder mode not supported")

            self.blocks = nn.ModuleList(decoder_modules)

        else:
            print("Decoder is normal transformer block")
            self.blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])
        
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x) # in:batch, x length +1, encoder_embed_dim
        #out: batch, length +1, decoder_embed_dim

        #append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        #add pos embed
        x = x + self.decoder_pos_embed

        if "swin" in self.decoder_mode: #!= 0 --> should be 0 in the original implementation
            B, L, D = x.shape # batch, length(patches), dim (decoder)
            x = x[:,1:,:]

    
        for blk in self.blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        #predictor projection
        pred = self.decoder_pred(x)

        if "swin" in self.decoder_mode:
            pred = pred
        else:
            pred = pred[:, 1:, :] # remove cls        

        return pred 
    

class AudioMAE(L.LightningModule):
    def __init__(self, 
                 norm_layer,
                 norm_pix_loss,
                 mask_ratio,
                 cfg_encoder,
                 cfg_decoder,
                 optimizer,
                 scheduler,
                 loss
    ):
        super().__init__()
        self.save_hyperparameters()

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio        
        self.optimizer_cfg = optimizer.target
        self.scheduler_cfg = scheduler 
        self.loss = hydra.utils.instantiate(loss)
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.target_length = cfg_encoder.img_size_x

        self.encoder = MAE_Encoder(
            img_size_x=cfg_encoder.img_size_x,
            img_size_y=cfg_encoder.img_size_y,
            patch_size=cfg_encoder.patch_size,
            in_chans=cfg_encoder.in_chans,
            embed_dim=cfg_encoder.embed_dim,
            depth=cfg_encoder.depth,
            num_heads=cfg_encoder.num_heads,
            mlp_ratio=cfg_encoder.mlp_ratio,
            norm_layer=norm_layer,
            pos_trainable=cfg_encoder.pos_trainable,
        )

        self.decoder = MAE_Decoder(
            embed_dim=cfg_encoder.embed_dim,
            decoder_embed_dim=cfg_decoder.embed_dim,
            num_patches=self.encoder.patch_embed.num_patches,
            decoder_depth=cfg_decoder.depth,
            decoder_num_heads=cfg_decoder.num_heads,
            mlp_ratio=cfg_decoder.mlp_ratio,
            norm_layer=norm_layer,
            patch_size=cfg_decoder.patch_size,
            in_chans=cfg_encoder.in_chans,
            decoder_mode=cfg_decoder.mode,
            pos_trainable=cfg_decoder.pos_trainable,
            no_shift=cfg_decoder.no_shift,
            target_length=self.target_length
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize encoder positional embeddings
        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.encoder.pos_embed.shape[-1], # embedding dim
            self.encoder.patch_embed.patch_hw, # 8,32
            cls_token=True
        )
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize decoder positional embeddings
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.decoder.decoder_pos_embed.shape[-1], # embedding_dim
            self.encoder.patch_embed.patch_hw,  # 8,32
            cls_token=True
        )
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def patchify(self, imgs):
        p = self.encoder.patch_embed.patch_size[0]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        
        return x

    def unpatchify(self, x): # changed? 
        p = self.encoder.patch_embed.patch_size[0]    
        h = self.target_length//p
        w = 128//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        #loss = self.loss(pred, target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #loss = self.loss(pred, target)
        #loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss   

    def forward(self, imgs):
        latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss_recon = self.forward_loss(imgs, pred, mask)
        return loss_recon, pred, mask

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        #labels = batch["label"]
        loss, pred, mask = self(audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size * self.trainer.num_nodes
        print("base learning rate on 256 was:", self.optimizer_cfg["lr"])
        self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 256
        print("effective learning rate now:", self.optimizer_cfg["lr"], self.layer_decay)
        
        param_groups = param_groups_weight_decay(
            self,
            self.optimizer_cfg["weight_decay"],
            no_weight_decay_list=("bias", "bn", "ln", "gn", "norm")
        )

        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.optimizer_cfg["lr"], 
            betas=self.optimizer_cfg["betas"])
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.09375 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1
            }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": optimizer}


    # def on_train_batch_start(self, batch, batch_idx, unused=0):
    #     # Log the learning rate
    #     lr = self.optimizers().param_groups[0]['lr']
    #     self.log('learning_rate', lr, prog_bar=True)


class AudioMAE_FT(L.LightningModule):
    def __init__(self, 
                 norm_layer,
                 cfg_encoder,
                 optimizer,
                 scheduler
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MAE_Encoder(
            img_size_x=cfg_encoder.img_size_x,
            img_size_y=cfg_encoder.img_size_y,
            patch_size=cfg_encoder.patch_size,
            in_chans=cfg_encoder.in_chans,
            embed_dim=cfg_encoder.embed_dim,
            depth=cfg_encoder.depth,
            num_heads=cfg_encoder.num_heads,
            mlp_ratio=cfg_encoder.mlp_ratio,
            norm_layer=norm_layer,
            pos_trainable=cfg_encoder.pos_trainable,
            stride=None
        )

        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler 

    def forward(self, audio):
        return self.encoder(audio)

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        loss  = self.loss(pred, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, 
            params=self.parameters())
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.05 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1
            }

            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": optimizer}


    def on_train_batch_start(self, batch, batch_idx, unused=0):
        # Log the learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, prog_bar=True)


class VIT(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 loss,
                 metric_cfg,
                 mask_t_prob,
                 mask_f_prob,
                 mask2d,
                 ema_update_rate,
                 mask_inference
    ):
        
        L.LightningModule.__init__(self)
        
        if mask_inference:
            num_classes = 9735 # XCL overwrite 

        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)
        self.mask_2d = mask2d

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        if self.global_pool == "attentive":
            #attentive_heads = self.embed_dim // self.num_heads
            #self.attentive_probe = AttentivePooling(self.embed_dim, self.num_heads)
            self.attentive_probe = AttentivePooling(dim=self.embed_dim, num_heads=self.num_heads)

        self.mask_2d = mask2d
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        metric = hydra.utils.instantiate(metric_cfg)
        
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
        self.ema = None
        if self.ema_update_rate: 
            self.ema = EMA(self, decay=ema_update_rate)

        self.class_mask = None
        if mask_inference:
            print("Logit Masking")
            hf_path = "DBD-research-group/BirdSet"
            hf_name = "XCL"
            pretrain_labels = datasets.load_dataset_builder(
                hf_path, hf_name, trust_remote_code=True).info.features["ebird_code"]
            inference_labels = datasets.load_dataset_builder(
                hf_path, mask_inference, trust_remote_code=True).info.features["ebird_code"]
            self.class_mask = [pretrain_labels.names.index(i) for i in inference_labels.names]
        


    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????

        if self.global_pool != "attentive": # everything except attentive is global
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.global_pool =="attentive":
            outcome = self.attentive_probe(x)
            outcome = self.fc_norm(outcome)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
    def forward_features_mask(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange

        if self.mask_2d: 
            x, mask, ids_restore = self.random_masking_2d(x)
        else:
            pass

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome 

    def forward(self, x):
        if self.mask_t_prob > 0.0 or self.mask_f_prob > 0.0: #shape val: 64, 1, 512, 128
            x = self.forward_features_mask(x)
        else:
            x = self.forward_features(x)
        pred = self.head(x)
        return pred 

    def random_masking_2d(self, x):
        N, L, D = x.shape
        T = 64 # AUDIOSET
        F = 8 # AUDIOSET

        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - self.mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - self.mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        #metric = self.val_metric(pred, targets)
        #pred = torch.softmax(pred, dim=1)
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema:
            self.ema.restore()
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=True)

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        if self.class_mask: 
            # if targets.shape == pred.shape:
            #     targets = targets[:, self.class_mask]
            pred = pred[:, self.class_mask]

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.restore()
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 48
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        if self.layer_decay:
            params = param_groups_lrd(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
                no_weight_decay_list=self.no_weight_decay(),
                layer_decay=self.layer_decay, #scaling favtor for ech layer 0.75^layer ..--> 0.75^0
                decay_type=self.decay_type
            )

            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params
            )

        else:
            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params=self.parameters())
            # print("LAMB")
            # self.optimizer = LAMB(self.parameters(), lr=3e-4)
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer=self.optimizer,
            #     num_warmup_steps=num_warmup_steps,
            #     num_training_steps=num_training_steps
            # )

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            self.load_state_dict(pretrained_state_dict, strict=False)

            trunc_normal_(self.head.weight, std=2e-5)

import copy
import torch

class EMA:
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.shadow_buffers = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.detach().clone()
                    self.shadow_params[name].requires_grad = False
            for name, buffer in model.named_buffers():
                self.shadow_buffers[name] = buffer.detach().clone()

        # We'll store the base model's original weights here (for restore)
        self.backup_params = {}
        self.backup_buffers = {}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = self.shadow_params[name].to(param.device)
                self.shadow_params[name].copy_(
                    self.decay * self.shadow_params[name] + (1.0 - self.decay) * param.data
                )
        for name, buffer in self.model.named_buffers():
            self.shadow_buffers[name] = buffer.detach().clone().to(buffer.device)

    @torch.no_grad()
    def apply_shadow(self):
        """
        1) Backup the base model's current params/buffers
        2) Load the shadow (EMA) weights into the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # backup base weights
                self.backup_params[name] = param.detach().clone()
                # load shadow
                param.copy_(self.shadow_params[name].to(param.device))

        for name, buffer in self.model.named_buffers():
            # backup base buffers
            self.backup_buffers[name] = buffer.detach().clone()
            # load shadow
            buffer.copy_(self.shadow_buffers[name].to(buffer.device))

    @torch.no_grad()
    def restore(self):
        """
        Restore the base model weights from `backup_...`
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.copy_(self.backup_params[name].to(param.device))
        for name, buffer in self.model.named_buffers():
            if name in self.backup_buffers:
                buffer.copy_(self.backup_buffers[name].to(buffer.device))

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        # Store initial lr, min_lr, and lr_scale for each param group
        self.init_lrs = []
        self.min_lrs = []
        self.lr_scales = []
        for param_group in optimizer.param_groups:
            self.init_lrs.append(param_group.get('initial_lr', param_group['lr'])) #could be kept for later use when doing per group lrs
            self.min_lrs.append(param_group.get('min_lr', self.min_lr)) # could be kept for later use when doing per group lrs
            self.lr_scales.append(param_group.get('lr_scale', 1.0))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(0, self.last_epoch)
        lrs = []
        for idx, (init_lr, min_lr, lr_scale) in enumerate(zip(self.init_lrs, self.min_lrs, self.lr_scales)):
            if step < self.warmup_steps:
                lr = init_lr * step / float(max(1, self.warmup_steps))
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                lr = min_lr + (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
            lr *= lr_scale
            lrs.append(lr)
        return lrs
        
from transformers import AutoConfig, ConvNextForImageClassification

class ConvNext(L.LightningModule):
    """
    ConvNext model for audio classification.
    """

    def __init__(
        self,
        num_channels,
        num_classes,
        optimizer,
        scheduler,
        hf_checkpoint, 
        loss,
        metric_cfg,
        model_dir
    ):
  
        super().__init__()
        self.save_hyperparameters()
        self.hf_checkpoint = hf_checkpoint
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.model_dir = model_dir

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        metric = hydra.utils.instantiate(metric_cfg)
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []


        if self.hf_checkpoint: 
            self.model = ConvNextForImageClassification.from_pretrained(
                self.hf_checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.model_dir,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            self.model = ConvNextForImageClassification(config)
    

    def forward(self, x):
        output = self.model(x)
        logits = output.logits

        return logits

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        pred = torch.sigmoid(pred)
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        pred = torch.sigmoid(pred)
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 256
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        if self.layer_decay:
            params = param_groups_lrd(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
               # no_weight_decay_list=self.no_weight_decay(), was ist das überhaupt
                layer_decay=self.layer_decay, #scaling favtor for ech layer 0.75^layer ..--> 0.75^0
                decay_type=self.decay_type
            )

            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params
            )

        else:
            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params=self.parameters())
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer=self.optimizer,
            #     num_warmup_steps=num_warmup_steps,
            #     num_training_steps=num_training_steps
            # )

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      
    
from models_ppnet import PPNet
#from models_ppnet import PPNetWithAttentivePrototype as PPNet

class VIT_ppnet(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 loss,
                 metric_cfg,
                 mask_t_prob,
                 mask_f_prob,
                 mask2d,
                 ema_update_rate,
                 ppnet_cfg,
                 mask_inference
    ):
        
        L.LightningModule.__init__(self)

        if mask_inference:
            num_classes = 411 # XCL overwrite 
            ppnet_cfg.num_classes = num_classes
        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()

        self.ppnet_cfg = ppnet_cfg
        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    #   for p in model.backbone_model.parameters():
    #     p.requires_grad = False
    # for p in model.add_on_layers.parameters():
    #     p.requires_grad = True
    # model.prototype_vectors.requires_grad = True      
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)
        self.mask_2d = mask2d

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        self.mask_2d = mask2d
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        metric = hydra.utils.instantiate(metric_cfg)
        
        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
        
        self.ema = None
        if self.ema_update_rate: 
            self.ema = EMA(self, decay=ema_update_rate)

        
        self.class_mask = None
        if mask_inference:
            print("Logit Masking")
            hf_path = "DBD-research-group/BirdSet"
            hf_name = "XCM"
            pretrain_labels = datasets.load_dataset_builder(
                hf_path, hf_name, trust_remote_code=True).info.features["ebird_code"]
            inference_labels = datasets.load_dataset_builder(
                hf_path, mask_inference, trust_remote_code=True).info.features["ebird_code"]
            self.class_mask = [pretrain_labels.names.index(i) for i in inference_labels.names]
        
        del self.head
        #del self.norm
        del self.fc_norm
        del self.head_drop
        

    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????
        x = self.norm(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = x[:, 0, :]
            x_patch = x[:, 1:, :] 
            z_f = x_patch - x_cls.unsqueeze(1) 
            try:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 32)
            except:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 64) # audioset
        else:
            x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 32)

        logits,_ = self.ppnet(x)

        return logits
    

    def forward(self, x):
        logits = self.forward_features(x)
        return logits 


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        logits = self(audio)
        targets = targets.long()
        #preds = logits.sigmoid()
        bce_loss = self.loss(logits, targets.float())
        orthogonality_loss = self.calculate_orthogonality_loss()

        self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('orthogonality_loss', orthogonality_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = bce_loss + orthogonality_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        #metric = self.val_metric(pred, targets)
        #pred = torch.softmax(pred, dim=1)
        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema:
            self.ema.restore()
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=True)

        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        if self.class_mask: 
        # if targets.shape == pred.shape:
        #     targets = targets[:, self.class_mask]
            pred = pred[:, self.class_mask]

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.restore()
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        
        from util.lr_decay import param_groups_lrd_pp
        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 48
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        if self.layer_decay:
            params = param_groups_lrd_pp(
                model=self,
                weight_decay=self.optimizer_cfg["weight_decay"],
                no_weight_decay_list=self.no_weight_decay(),
                layer_decay=self.layer_decay, #scaling favtor for ech layer 0.75^layer ..--> 0.75^0
                decay_type=self.decay_type,
                last_layer_lr=self.ppnet_cfg.last_layer_lr,
                prototype_lr=self.ppnet_cfg.prototype_lr,
            )

            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                params
            )

        else:
            print("TEST:",self.ppnet_cfg.last_layer_lr)
            # self.optimizer = hydra.utils.instantiate(
            #     self.optimizer_cfg, 
            #     params=self.parameters())
            optimizer_specifications = []

            #1) Add the add_on_layers group
            addon_params = list(self.ppnet.add_on_layers.parameters())
            optimizer_specifications.append({
                "params": addon_params,
                "lr": 3e-2,
                "weight_decay": 1e-4,
            })

            # 2) Add the prototype_vectors group
            #    (assuming this is either a list of Tensors or just one Tensor)
            proto_params = [self.ppnet.prototype_vectors]  # or list(...)
            optimizer_specifications.append({
                "params": proto_params,
                "lr": self.ppnet_cfg.prototype_lr,
            })

            # 3) Add the last_layer group
            last_params = list(self.ppnet.last_layer.parameters())
            optimizer_specifications.append({
                "params": last_params,
                "lr": self.ppnet_cfg.last_layer_lr,
                "weight_decay": 1e-4,
            })

            # 4) If there are truly "rest" parameters:
            all_params = set(self.parameters())
            already_in_groups = set(addon_params + proto_params + last_params)
            rest = [p for p in all_params if p not in already_in_groups]
            if len(rest) > 0:
                optimizer_specifications.append({"params": rest})

            # 5) Instantiate via Hydra
            self.optimizer = hydra.utils.instantiate(
                self.optimizer_cfg, 
                optimizer_specifications
            )
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            # scheduler = get_cosine_schedule_with_warmup(
            #     optimizer=self.optimizer,
            #     num_warmup_steps=num_warmup_steps,
            #     num_training_steps=num_training_steps
            # )

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            if "encoder_ema.cls_token" not in pre_state_dict: # without mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        new_key = key[len("encoder."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            else: # with mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        continue
                    elif key.startswith("projectors."):
                        continue
                    elif key.startswith("predictors."):
                        continue
                    elif key.startswith("encoder_ema."):
                        # Remove the "encoder_ema." prefix
                        new_key = key[len("encoder_ema."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            for k in ['head.weight', 'head.bias']:
                if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            info = self.load_state_dict(pretrained_state_dict, strict=False)

            try:
                trunc_normal_(self.head.weight, std=2e-5)
            except:
                print("no head")

            # try: 
            #     trunc_normal_(self.ppnet.last_layer.weight, std=2e-5)
            # except:
            #     print("no prototype vectors")

    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss



class VIT_MIM(L.LightningModule):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 pretrained_weights_path, 
                 target_length,
                 mim_cfg,
                 optimizer_cfg
    ):
        L.LightningModule.__init__(self)

        self.encoder = VisionTransformer(
            img_size = (img_size_x, img_size_y),
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = 1,
            drop_path_rate=drop_path,
        )
        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length

        self.load_pretrained_weights(pretrained_weights_path, "XCL")
        
        #self.encoder.load_pretrained_weights(pretrained_weights_path, "XCL")

        self.encoder_ema = copy.deepcopy(self.encoder)
        for p in self.encoder_ema.parameters():
            p.requires_grad = False

        #del self.encoder.head
        # del self.encoder.norm
        # del self.encoder.fc_norm
        # del self.encoder.head_drop

        # del self.encoder_ema.norm
        # del self.encoder_ema.fc_norm
        # del self.encoder_ema.head_drop

        
        proj_dim = mim_cfg.proj_dim
        out_dim = mim_cfg.out_dim
        pred_dim = mim_cfg.pred_dim
        self.queue_size = mim_cfg.queue_size
        self.momentum = mim_cfg.momentum
        self.temperature = mim_cfg.temperature

        # number of last layers to modify based on total depth
        if depth == 24: # Large
            modify_last_n = 8 # MIM-Refiner
        elif depth == 32: # Huge
            modify_last_n = 12 # MIM-Refiner
        elif depth == 12: # Base
            modify_last_n = 4 # hmm let's experiment with 4 or 6 ?
        else:
            modify_last_n = int(0.35*depth) # default, maybe last 35% of layers?

        self.modify_last_n = modify_last_n
        self.start_modify = depth - self.modify_last_n

        # Create MLP projectors for the last N layers
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, proj_dim, bias=False), # don't use bias as it is followed by BN
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim, affine=False),
            ) for _ in range(self.modify_last_n)
        ])

        self.predictors = nn.ModuleList([
            Predictor(hidden_dim=pred_dim, out_dim=out_dim) for _ in range(self.modify_last_n)
        ]) 

        # load pretrained weights here 
        # second step: 20 epochs with different learning rate, 30 for mim update. 

        self.save_hyperparameters()
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias 
        self.optimizer_cfg = optimizer_cfg

        

        self.queues = [torch.randn(self.queue_size, out_dim).to("cuda") for _ in range(self.modify_last_n)]

        

    # def forward_features(self, x):
    #     B = x.shape[0]
    #     x = self.patch_embed(x) # batch, patch, embed
    #     x = x + self.pos_embed[:, 1:, :] 
    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(B, -1, -1) 
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = self.pos_drop(x)        

    #     for blk in self.blocks:
    #         x = blk(x)

    #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #     outcome = self.fc_norm(x)


    #     return outcome

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     pred = self.head(x)
    #     return pred 


    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :] 
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder.pos_drop(x)      
        # # pre-transformer layers
        # x = self.vit.to_patch_embedding(x)
        # x = self.vit.dropout(x)

        # transformer layers
        zs = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)

            if i >= self.start_modify:
                #intermediate_x = x[:,0] #cls_token
                intermediate_x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                intermediate_x = self.fc_norm(intermediate_x)
                # projector
                z = self.projectors[i - self.start_modify](intermediate_x)
                zs.append(z)

        # # post-transformer layers
        # x = self.encoder.to_latent(x)
        # x = self.vit.mlp_head(x)

        return zs


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        zs = self(audio)

        contrastive_loss = 0 

        NN_zs = []
        for idx, (z, queue, predictor) in enumerate(zip(zs, self.queues, self.predictors)):
            NN_z = self.NN(z, queue)
            NN_zs.append(NN_z) # retrieve nearest neighbor for each layer
            self.queues[idx] = self.update_queue(queue, z) # update queue for each layer

            h = predictor(z)
            contrastive_loss += loss_fn(NN_z, h, self.temperature)

        contrastive_loss /= len(zs) # average over layers, maybe give different weights to different layers?

        self.update_ema(self.encoder, self.encoder_ema, self.momentum) # keeping track of EMA for downstream tasks

        self.log("train_contrastive_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True)

        return contrastive_loss
    
    def update_queue(self, queue, new_embeddings):
        return torch.cat((queue, new_embeddings.detach()), dim=0)[-self.queue_size:]  # Keep only the most recent `queue_size` entries
    
    def update_ema(self, model, ema_model, decay):
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                ema_param.data = decay * ema_param.data + (1 - decay) * param.data

    def NN(self, key, Queue):
        # Nearest Neighbor function to retrieve the positive in the queue
        key = F.normalize(key, dim=1)
        Queue = F.normalize(Queue, dim=1)
        similarity = torch.mm(key, Queue.t())
        nearest_neighbors = similarity.max(dim=1)[1]
        return Queue[nearest_neighbors]

    def configure_optimizers(self):
        #heuristic:
        # eff_batch_size = self.trainer.accumulate_grad_batches * self.trainer.num_devices * self.train_batch_size
        # self.optimizer_cfg["lr"] = self.optimizer_cfg["lr"] * eff_batch_size / 48
        # print("effective learning rate:", self.optimizer_cfg["lr"], self.layer_decay)

        params = list(self.encoder.parameters())

        for i in range(len(self.predictors)):
            params += list(self.projectors[i].parameters()) + list(self.predictors[i].parameters())
        
        self.optimizer = torch.optim.AdamW(
            lr=self.optimizer_cfg.target["lr"],
            weight_decay=self.optimizer_cfg.target["weight_decay"],
            betas=(0.9, 0.95),
            params=params
        )
    
        num_training_steps = self.trainer.estimated_stepping_batches
        warmup_ratio = 0.2 # hard coded
        num_warmup_steps = num_training_steps * warmup_ratio

        scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=num_warmup_steps,
            total_steps=num_training_steps
        )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",  # Update at every step
            "frequency": 1,
            "name": "lr_cosine"
        }

        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.encoder.patch_embed = PatchEmbed(img_size, 16, 1, self.encoder.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.encoder.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value
            
            info = self.encoder.load_state_dict(pretrained_state_dict, strict=False)

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.encoder.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.encoder.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.encoder.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            self.encoder.load_state_dict(pretrained_state_dict, strict=False)

            trunc_normal_(self.head.weight, std=2e-5)


#### BirdAVES, download cfg and weights from : https://github.com/earthspecies/aves (table in the bottom)
from torchaudio.models import wav2vec2_model

class BirdAVES(L.LightningModule, nn.Module):

    def __init__(self,
                 num_classes,
                 optimizer,
                 scheduler,
                 loss,
                 metric_cfg,
                 birdaves_cfg_path,
                 birdaves_weights_path,
    ):

        L.LightningModule.__init__(self)

        birdaves_cfg = birdaves_cfg_path
        self.encoder = wav2vec2_model(**birdaves_cfg, aux_num_out=None)
        self.encoder.load_state_dict(torch.load(birdaves_weights_path))
        self.encoder.feature_extractor.requires_grad_(False) # feature extractor should never be retrained

        self.embed_dim = 768 if 'base' in birdaves_weights_path else 1024
        self.head = nn.Linear(self.embed_dim, num_classes)

        self.save_hyperparameters()

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        metric = hydra.utils.instantiate(metric_cfg)

        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        features = self.encoder.extract_features(x.squeeze())[0][-1]
        pred = self.head(features.mean(dim=1))
        return pred

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=False)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=False)

        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        params = []
        params += list(self.head.parameters())
        params += list(self.encoder.parameters())

        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=params)

        if self.scheduler_cfg:
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}

class BirdAVES_ppnet(BirdAVES):
    def __init__(self, ppnet_cfg, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.ppnet_cfg = ppnet_cfg

        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    def forward(self, x):
        features = self.encoder.extract_features(x.squeeze())[0][-1] # 64, 249, 1024 # batch, token, embed
        if self.ppnet_cfg.focal_similarity == True:
            x_pooled = features.mean(dim=1) # 64, 1024 # batch, embed
            z_f = features - x_pooled.unsqueeze(1) # 64, 249, 1024 # batch, token, embed

            features = z_f.permute(0, 2, 1).unsqueeze(2) # 64, 1024, 249 -> batch, embed, 1, token
        else:
            features = features.permute(0, 2, 1).unsqueeze(2) #
        logits, _ = self.ppnet(features)

        return logits

    def configure_optimizers(self):
        optimizer_specifications = []

        # 1) Add the add_on_layers group
        addon_params = list(self.ppnet.add_on_layers.parameters())
        optimizer_specifications.append({
            "params": addon_params,
            "lr": 3e-2,
            "weight_decay": 1e-4,
        })

        # 2) Add the prototype_vectors group
        #    (assuming this is either a list of Tensors or just one Tensor)
        proto_params = [self.ppnet.prototype_vectors]  # or list(...)
        optimizer_specifications.append({
            "params": proto_params,
            "lr": self.ppnet_cfg.prototype_lr,
        })

        # 3) Add the last_layer group
        last_params = list(self.ppnet.last_layer.parameters())
        optimizer_specifications.append({
            "params": last_params,
            "lr": self.ppnet_cfg.last_layer_lr,
            "weight_decay": 1e-4,
        })

        # 4) If there are truly "rest" parameters:
        all_params = set(self.parameters())
        already_in_groups = set(addon_params + proto_params + last_params)
        rest = [p for p in all_params if p not in already_in_groups]
        if len(rest) > 0:
            optimizer_specifications.append({"params": rest})

        # 5) Instantiate via Hydra
        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            optimizer_specifications
        )

        if self.scheduler_cfg:
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}

#### SimCLR
#from https://huggingface.co/ilyassmoummad/ProtoCLR
from ProtoCLR.cvt import ConvolutionalVisionTransformer
from ProtoCLR.melspectrogram import MelSpectrogramProcessor

class SimCLR(L.LightningModule, nn.Module):

    def __init__(self,
                 num_classes,
                 optimizer,
                 scheduler,
                 loss,
                 metric_cfg,
                 model_spec_cfg,
                 proto_clr_weights_path,
    ):

        L.LightningModule.__init__(self)
        self.cuda()
        self.preprocessor = MelSpectrogramProcessor(device=self.device)
        self.encoder = ConvolutionalVisionTransformer(spec=model_spec_cfg)
        self.encoder.load_state_dict(torch.load(proto_clr_weights_path, map_location="cpu"))
        self.encoder.cuda()

        self.embed_dim = 384
        self.head = nn.Linear(self.embed_dim, num_classes)

        self.save_hyperparameters()

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        metric = hydra.utils.instantiate(metric_cfg)

        additional_metrics = []
        if metric_cfg.get("additional"):
            for _, metric_cfg in metric_cfg.additional.items():
                additional_metrics.append(hydra.utils.instantiate(metric_cfg))
        add_metrics = MetricCollection(additional_metrics)
        self.test_add_metrics = add_metrics.clone()
        self.val_add_metrics = add_metrics.clone()

        self.train_metric = metric.clone()
        self.val_metric = metric.clone()
        self.test_metric = metric.clone()

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocessor.process(x)
        features = self.encoder(x)
        pred = self.head(features)
        return pred

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)
        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.val_predictions.append(pred.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

        #self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_predictions)
        targets = torch.cat(self.val_targets)
        metric = self.val_metric(preds, targets)
        self.log(f'val_{self.val_metric.__class__.__name__}', metric, on_step=False, on_epoch=True, prog_bar=False)
        print("val metric:", metric.detach().cpu().item())

        self.val_add_metrics(preds, targets)
        for name, metric in self.val_add_metrics.items():
            self.log(f'valid_{name}', metric, on_epoch=True, prog_bar=False)

        self.val_predictions = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        pred = self(audio)

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())

        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_predictions)
        targets = torch.cat(self.test_targets)
        self.test_metric(preds, targets)
        self.log(f'test_{self.test_metric.__class__.__name__}', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_add_metrics(preds, targets)
        for name, metric in self.test_add_metrics.items():
            self.log(f'test_{name}', metric, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        params = []
        params += list(self.encoder.parameters())
        params += list(self.head.parameters())

        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=params)

        if self.scheduler_cfg:
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}


class SimCLR_ppnet(SimCLR):
    def __init__(self, ppnet_cfg, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.ppnet_cfg = ppnet_cfg

        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.preprocessor.process(x)

        for i in range(self.encoder.num_stages):
            x, cls_token = getattr(self.encoder, f'stage{i}')(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = cls_token.squeeze().unsqueeze(-1).unsqueeze(-1) # [64, 384, 1, 1]
            x_patch = x # [64, 384, 8, 19]
            x = x_patch - x_cls
        else:
            x = x

        logits, _ = self.ppnet(x)

        return logits

    def configure_optimizers(self):
        optimizer_specifications = []

        # 1) Add the add_on_layers group
        addon_params = list(self.ppnet.add_on_layers.parameters())
        optimizer_specifications.append({
            "params": addon_params,
            "lr": 3e-2,
            "weight_decay": 1e-4,
        })

        # 2) Add the prototype_vectors group
        #    (assuming this is either a list of Tensors or just one Tensor)
        proto_params = [self.ppnet.prototype_vectors]  # or list(...)
        optimizer_specifications.append({
            "params": proto_params,
            "lr": self.ppnet_cfg.prototype_lr,
        })

        # 3) Add the last_layer group
        last_params = list(self.ppnet.last_layer.parameters())
        optimizer_specifications.append({
            "params": last_params,
            "lr": self.ppnet_cfg.last_layer_lr,
            "weight_decay": 1e-4,
        })

        # 4) If there are truly "rest" parameters:
        all_params = set(self.parameters())
        already_in_groups = set(addon_params + proto_params + last_params)
        rest = [p for p in all_params if p not in already_in_groups]
        if len(rest) > 0:
            optimizer_specifications.append({"params": rest})

        # 5) Instantiate via Hydra
        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            optimizer_specifications
        )

        if self.scheduler_cfg:
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.067  # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=num_warmup_steps,
                total_steps=num_training_steps
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}


class Predictor(nn.Module):
    def __init__(self, hidden_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)

def loss_fn(nn, p, temperature=0.1):
    nn = normalize(nn, dim=1)
    p = normalize(p, dim=1)
    logits = torch.matmul(nn, p.T)
    logits /= temperature
    labels = torch.arange(p.size(0), device=p.device)
    return F.cross_entropy(logits, labels)






# class AttentivePooling(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         """
#         Args:
#             embed_dim: Dimension of the patch embeddings.
#             num_heads: Number of attention heads.
#         """
#         super(AttentivePooling, self).__init__()
#         # Using torch's built-in multihead attention
#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         # Learnable query parameter, shape: (1, 1, embed_dim)
#         # This query will be repeated for each sample in the batch.
#         self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (B, N, embed_dim) where the first token is the CLS token.
#         Returns:
#             Pooled features of shape (B, embed_dim).
#         """
#         # Exclude the CLS token and work on patch tokens only.
#         x_patch = x[:, 1:, :]  # shape: (B, N-1, embed_dim)
#         B = x_patch.shape[0]
        
#         # Expand the learnable query for each instance in the batch.
#         # Query shape becomes: (B, 1, embed_dim)
#         query_expanded = self.query.expand(B, -1, -1)
        
#         # Apply multihead attention:
#         # Query: (B, 1, embed_dim)
#         # Key, Value: (B, N-1, embed_dim)
#         # Output shape: (B, 1, embed_dim)
#         attn_output, attn_weights = self.mha(query_expanded, x_patch, x_patch)
        
#         # Squeeze the sequence dimension to obtain (B, embed_dim)
#         pooled = attn_output.squeeze(1)
#         return pooled





import torch
import torch.nn as nn
import torch.nn.functional as F

# class AttentivePooling(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         """
#         Args:
#             embed_dim: Dimension of the patch embeddings, i.e. d.
#             num_heads: Number of attention heads.
#                        Must divide embed_dim evenly.
#         """
#         super(AttentivePooling, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#         self.d_h = embed_dim // num_heads  # per-head dimension
        
#         # For each head h, define:
#         # - A key projection matrix Wk_h of shape (d_h, embed_dim)
#         # - A value projection matrix Wv_h of shape (d_h, embed_dim)
#         # - A learnable query vector q_h of shape (d_h)
#         self.Wk = nn.Parameter(torch.randn(num_heads, self.d_h, embed_dim))
#         self.Wv = nn.Parameter(torch.randn(num_heads, self.d_h, embed_dim))
#         self.q = nn.Parameter(torch.randn(num_heads, self.d_h))
        
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (B, N, embed_dim). 
#                Typically, x should be the patch tokens (excluding any CLS token).
#         Returns:
#             Pooled features of shape (B, embed_dim).
#         """
#         B, N, d = x.shape  # d should equal embed_dim
        
#         # Compute keys for each head:
#         # x_keys: shape (B, num_heads, N, d_h)
#         x_keys = torch.einsum('hjd,bnd->bhnj', self.Wk, x)
        
#         # Compute values for each head:
#         # x_values: shape (B, num_heads, N, d_h)
#         x_values = torch.einsum('hjd,bnd->bhnj', self.Wv, x)
        
#         # Compute attention scores for each head:
#         # scores: shape (B, num_heads, N)
#         scores = torch.einsum('bhnj,hj->bhn', x_keys, self.q)
        
#         # Apply softmax over the patch dimension (N):
#         attn_weights = F.softmax(scores, dim=-1)  # shape (B, num_heads, N)
        
#         # Compute the weighted sum of values for each head:
#         # p_hat: shape (B, num_heads, d_h)
#         p_hat = torch.einsum('bhn,bhnj->bhj', attn_weights, x_values)
        
#         # Concatenate the outputs from all heads to form the final descriptor:
#         pooled = p_hat.reshape(B, -1)  # shape (B, num_heads * d_h) = (B, embed_dim)
#         return pooled


class AttentivePooling(nn.Module):
    # taken from OG paper: https://github.com/apple/ml-aim/blob/main/aim-v1/aim/v1/torch/layers.py
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        # self.bn = (
        #     nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        #     if use_batch_norm
        #     else nn.Identity()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:, :]  # exclude the ViT CLS token (could also be done in code later if neccessary)
        B, N, C = x.shape
        #x = self.bn(x.transpose(-2, -1)).transpose(-2, -1) #done with fc_norm later
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls
        return x_cls