#%%
import lightning as L
import torch
import torch.nn as nn
import hydra
import torchmetrics
from timm.models.vision_transformer import Block
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible
from util.patch_embed import PatchEmbed_new, PatchEmbed_org
from transformers import get_cosine_schedule_with_warmup

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
                 pos_trainable,
                 stride):
        super().__init__()
        # input: (Batch, Channel, Height, Width)
        # output: (Batch, #Patch, Embed)
        self.patch_embed = PatchEmbed_org((img_size_x, img_size_y), patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #norm_layer = nn.LayerNorm()
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
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
        x = self.patch_embed(x)

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
        ):
        super().__init__()
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)
        norm_layer = partial(nn.LayerNorm, eps=1e-6) # for both?

        
        if decoder_mode == "swin":
            decoder_modules = []
            window_size = (4,4)
            feat_size = (64,8)

            for i in range(16):
                if (i % 2) == 0:
                    shift_size = (0,0)
                else:
                    shift_size = (2,0)

                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
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

            self.blocks = nn.ModuleList(decoder_modules)

        else:
            self.blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])
        
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        #append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        #add pos embed
        x = x + self.decoder_pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        #predictor projection
        pred = self.decoder_pred(x)

        #remove cls token
        pred = pred[:, 1:, :]

        return pred 
    

class AudioMAE(L.LightningModule):
    def __init__(self, 
                 norm_layer,
                 norm_pix_loss,
                 cfg_encoder,
                 cfg_decoder,
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
            pos_trainable=cfg_decoder.pos_trainable
        )

        self.norm_pix_loss = norm_pix_loss
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler 

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize encoder positional embeddings
        pos_embed = get_2d_sincos_pos_embed_flexible(
            self.encoder.pos_embed.shape[-1], 
            self.encoder.patch_embed.patch_hw, 
            cls_token=True
        )
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize decoder positional embeddings
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
            self.decoder.decoder_pos_embed.shape[-1], 
            self.encoder.patch_embed.patch_hw, 
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

    def unpatchify(self, x):
        p = self.encoder.patch_embed.patch_size[0]    
        h = 1024//p
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

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss   

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        labels = batch["label"]
        loss, pred, mask = self(audio)
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


class VIT(L.LightningModule, VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 mask2d,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 loss):
        
        L.LightningModule.__init__(self)
        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y),
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes
        )
        self.save_hyperparameters()
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)
        self.mask_2d = mask2d

        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 

        self.loss = hydra.utils.instantiate(loss)
        self.optimizer = None
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=50) # hardcoded!!

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, :256, :] # strange
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

    def forward(self, x, v=None, mask_t_prob=0.0, mask_f_prob=0.0):
        x = self.forward_features(x)
        pred = self.head(x)
        return pred 


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()
        loss  = self.loss(pred, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        pred = self(audio)
        targets = targets.long()

        loss = self.loss(pred, targets)
        accuracy = self.accuracy(pred, targets)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, 
            params=self.parameters())
    
        if self.scheduler_cfg: 
            num_training_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = 0.05 # hard coded
            num_warmup_steps = num_training_steps * warmup_ratio

            scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # Update at every step
                "frequency": 1,
                "name": "lr_cosine"
            }

            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        
        return {"optimizer": self.optimizer}      


