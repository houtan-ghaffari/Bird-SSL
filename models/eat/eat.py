import math
import random
import json
from pathlib import Path
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
from sklearn import metrics
import librosa
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Spectrogram
from torchaudio.transforms import MelScale
from torchaudio.transforms import AmplitudeToDB
from torchaudio.transforms import TimeMasking
from torchaudio.transforms import FrequencyMasking
from torchaudio.transforms import TimeStretch
# from torchaudio.functional import pitch_shift
import torchvision

from datasets import load_dataset
from datasets import Dataset as HFD

device = 'cuda'
seed = 42

# Decoder
class Conv2dLayerNorm(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding='same', groups=1, activation=nn.GELU, add_residual=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups)
        self.add_residual = add_residual
        self.ln = nn.LayerNorm(out_c, elementwise_affine=False)
        self.activation = activation()
        
    def forward(self, x):
        inp = x
        x = self.conv(x)
        x = x.transpose(-3, -1)
        x = self.ln(x)
        x = x.transpose(-3, -1)
        x = self.activation(x)
        if self.add_residual and x.size(1) == inp.size(1):
            x = x + inp
        return x

class CNN2dDecoder(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding='same', groups=16, activation=nn.GELU, add_residual=True, num_layers=6, num_freq_patches=8, num_time_patches=32):
        super().__init__()
        self.blocks = nn.Sequential(*[Conv2dLayerNorm(dim, dim, kernel_size, stride, padding, groups, activation, add_residual) for _ in range(num_layers)])
        self.proj = nn.Linear(dim, dim, bias=True) # decoder to patch
        self.f = num_freq_patches
        self.t = num_time_patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore)
        x = x.transpose(-2, -1)  # B, D, L
        x = x.reshape(x.shape[0], x.shape[1], self.t, self.f)
        x = self.blocks(x)
        x = x.flatten(2).transpose(-2, -1)  # B, L, D
        return self.proj(x)

# Vit Encoder for EAT

## https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L170
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class FeedForward(nn.Module):

    def __init__(self, in_dim, hid_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid_dim, in_dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.ffn(x)

class Attention(nn.Module):

    def __init__(self, in_dim, num_heads=None, qkv_bias=True, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_dim, in_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.tau = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if use_tau else None

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # B, H, N, D @ B, H, D, N
        # if self.tau is not None:
            # attn = attn * self.tau
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v  # B, H, N, N @ B, H, N, D -> B, H, N, D
        x = x.transpose(1, 2).reshape(batch_size, seq_len, feat_dim)  # B, H, N, D -> B, N, H, D -> B, N, H * D
        x = self.proj(x)
        return self.proj_drop(x)

class EncoderBlock(nn.Module):

    def __init__(self, in_dim, num_heads, expand_ratio=4., qkv_bias=True, dropout=0, drop_path=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(in_dim, num_heads, qkv_bias)
        self.ff = FeedForward(in_dim, int(in_dim * expand_ratio), dropout)
        self.ln1 = norm_layer(in_dim)
        self.ln2 = norm_layer(in_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.ln1(x)))
        return x + self.drop_path2(self.ff(self.ln2(x)))

class PatchEmbed(nn.Module):

    def __init__(self, img_size=(512, 128), patch_size=(16, 16), in_chans=1, emb_dim=768):
        super().__init__()
        assert isinstance(patch_size, tuple)
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 256
        self.patch_ft = (img_size[1] // patch_size[1], img_size[0] // patch_size[0]) # number of patches height/width = 8/32
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
       
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x) # 1, 1, 512, 128 -> 1, 768, 32, 8 (batch, 768 channel, 32 height, 8 width)
        x = x.flatten(2) # 1, 768, 32, 8 -> 1, 768, 256
        x = x.transpose(1, 2) # 1, 768, 256 -> 1, 256, 768
        return x

class EAT_Encoder(nn.Module):
    def __init__(self, input_shape=(512, 128), patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, dropout=0., drop_path_rate=0., pos_trainable=False, clone_size=16, mode='student', mask_mode='inv'):
        super().__init__()
        
        assert mode in ['student', 'teacher']
        assert mask_mode in ['rand', 'inv']
        assert (input_shape[0] % patch_size[0]) == 0 and (input_shape[1] % patch_size[1]) == 0
        
        if mode == 'student':
            self.forward_fn = self.student_forward
            if mask_mode == 'rand':
                self.mask_fn = self.random_masking
            else:
                self.mask_fn = self.inverse_block_mask
        else:
            self.forward_fn = self.teacher_forward
            self.mask_fn = None

        self.clone_size = clone_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=pos_trainable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1, 1, 768
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, dpr[i], norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def inverse_block_mask(self, shape, mask_ratio=0.8, num_freq_patches=8, num_time_patches=32, mask_length=5, mask_prob_adjust=0.1, require_same_masks=True):
        
        if mask_ratio == 0:
            return x, None, None
        
        assert mask_length > 1
        
        B, L, D = shape
        B = B * self.clone_size
        d = (num_time_patches, num_freq_patches)
        mask_ratio = 1 - mask_ratio    
        
        mask = torch.zeros((B, d[0], d[1]))
        masking_size = int(L * ((mask_ratio + mask_prob_adjust) / (mask_length ** 2)))
        mask_inds = torch.randint(0, L, size=(B, masking_size))
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        
        centers = mask.nonzero(as_tuple=True)
        inds = ([], [], [])
        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)
        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)
        mask[(i0, i1, i2)] = 1
        mask = mask.reshape(B, -1)
        
        if require_same_masks:
            n_masks = mask.sum(dim=-1)
            target_len = int(L * (mask_ratio))
            for i in range(len(mask)):
                n = n_masks[i]
                m = mask[i]
                r = 0
                if n > target_len:
                    to_unmask = torch.multinomial(m, int(n - target_len), replacement=False)
                    m[to_unmask] = 0
                elif n < target_len:
                    to_mask = torch.multinomial((1 - m), int(target_len - n), replacement=False)
                    m[to_mask] = 1
                    
        # now inverse_mask: 1 are places to remove, 0 are places to keep.
        mask = 1 - mask  
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)
        len_keep = L - mask[0].sum()
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        return mask.float(), ids_keep, ids_restore
    
    def random_masking(self, shape, mask_ratio=0.8, *args):
        
        if mask_ratio == 0:
            return x, None, None
        
        B, L, D = shape  # batch, length, dim
        B *= self.clone_size
        
        len_keep = int(L * (1 - mask_ratio))    
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
        ids_restore = ids_shuffle.argsort(dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, D)
        ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        
        return mask, ids_keep, ids_restore

    def student_forward(self, x, mask_ratio=0.8):
        x = self.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        x = x + self.pos_embed[:, 1:, :]
        
        # generate masks of shape: (B*clone_size, L) 
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        mask, ids_keep, ids_restore = self.mask_fn(x.shape, mask_ratio, num_freq_patches, num_time_patches)
        mask, ids_keep, ids_restore = mask.to(x.device), ids_keep.to(x.device), ids_restore.to(x.device)

        # repeat the inputs for clone_size on batch axis
        x = x.repeat_interleave(self.clone_size, dim=0)

        # mask the input
        x = torch.gather(x, dim=1, index=ids_keep)  # B * clone_size, L * (1 - mask_ratio), D
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # separate cls from the rest
        cls_pred = x[:, 0]
        patch_pred = x[:, 1:]
        
        return cls_pred, patch_pred, mask, ids_restore

    def teacher_forward(self, x, mask_ratio=0):
        x = self.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x[:, 1:, :].clone())
        return features
       
    def forward(self, x, mask_ratio=0.8): 
        return self.forward_fn(x, mask_ratio)

# Student model for EAT
class EAT_Student(nn.Module):
    
    def __init__(self,
                 input_shape=(512, 128), 
                 patch_size=(16, 16),
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=16,
                 mask_mode='inv',
                 decoder_cls=CNN2dDecoder,
                 decoder_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6},
                ):
        
        super().__init__()

        # student encoder & decoder
        self.encoder = EAT_Encoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, dropout, drop_path_rate, pos_trainable, clone_size, mode='student', mask_mode=mask_mode)
        num_freq_patches, num_time_patches = self.encoder.patch_embed.patch_ft
        self.decoder = decoder_cls(embed_dim, num_freq_patches=num_freq_patches, num_time_patches=num_time_patches, **decoder_kwargs)
        self.initialize_weights()
        self.clone_size = clone_size
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.encoder.pos_embed.shape[-1], self.encoder.patch_embed.patch_ft, cls_token=True)
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.LayerNorm):
            # nn.init.constant_(m.bias, 0)
            # nn.init.constant_(m.weight, 1.0)
    
    
    def forward(self, x, mask_ratio=0.8):
        """
        args:
            x - input mel-spectrogram of shape B, 1, T, F 
        """
        cls_pred, patch_pred, mask, ids_restore = self.encoder(x, mask_ratio)
        patch_pred = self.decoder(patch_pred, ids_restore)
        return cls_pred, patch_pred, mask

# Teacher model for EAT
class EAT_Teacher(nn.Module):
    
  def __init__(self,
                 input_shape=(512, 128), 
                 patch_size=(16, 16),
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=16,
                 average_top_k_layers=12,
                 instance_norm_target_layer=True,  # based on EAT config
                 batch_norm_target_layer=False,  # based on EAT config
                 layer_norm_target_layer=False,  # based on EAT config
                 layer_norm_targets=True,  # based on EAT config
                 instance_norm_targets=False,  # based on EAT config
                ):
        
        super().__init__()

        self.encoder = EAT_Encoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, dropout, drop_path_rate, pos_trainable, clone_size, mode='teacher')
        # self.num_freq_patches, self.num_time_patches = self.encoder.patch_embed.patch_ft
        self.clone_size = clone_size
        self.average_top_k_layers = average_top_k_layers
        self.instance_norm_target_layer = instance_norm_target_layer
        self.batch_norm_target_layer = batch_norm_target_layer
        self.layer_norm_target_layer = layer_norm_target_layer
        self.layer_norm_targets = layer_norm_targets
        self.instance_norm_targets = instance_norm_targets

    
    def make_targets(self, y):   
        y = y[-self.average_top_k_layers:]
        permuted = False
        if self.instance_norm_target_layer or self.batch_norm_target_layer:  # BTC -> BCT
            y = [tl.transpose(1, 2) for tl in y]
            permuted = True
        if self.batch_norm_target_layer:
            y = [F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True) for tl in y]
        if self.instance_norm_target_layer:
            y = [F.instance_norm(tl.float()) for tl in y]
        if permuted: # BCT -> BTC
            y = [tl.transpose(1, 2) for tl in y]
        if self.layer_norm_target_layer:
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
    
        y = torch.stack(y).mean(dim=0)  # average layers' outputs
        if self.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])
        if self.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x):
        """
        args:
            x - input mel-spectrogram of shape B, 1, T, F 
        """
        patch_target = self.encoder(x, 0)  # outputs a list of all transformer layers' embeddings (e.g., 12 layer in base vit)
        patch_target = self.make_targets(patch_target)
        patch_target = patch_target.repeat_interleave(self.clone_size, dim=0)
        return patch_target

# Utils
class RiseRunDecay(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, steps_in_epoch=None, warmup=10, constant=0, total_epochs=None, lowest_lr=1e-6):

        self.warmup = warmup * steps_in_epoch
        self.constant = self.warmup + (constant * steps_in_epoch)
        self.final_step = total_epochs * steps_in_epoch
        self.decay_interval = self.final_step - self.constant
        self.lowest_lr = lowest_lr
        super().__init__(optimizer)

    def get_lr(self):
        current_iteration = self.last_epoch
        if current_iteration <= self.warmup:
            factor = current_iteration / self.warmup
        elif current_iteration <= self.constant:
            factor = 1.0
        else:
            current_iteration = self.last_epoch - self.constant
            factor = 0.5 * (1 + math.cos(math.pi * current_iteration / self.decay_interval))

        return [lr * factor if (lr * factor) > self.lowest_lr else self.lowest_lr for lr in self.base_lrs]

class EMA_Weight_Decay_Scheduler:
    def __init__(self, decay_start=0.9998, decay_end=0.99999, max_iter=None):
        self.decays = np.linspace(decay_start, decay_end, max_iter, dtype=np.float32).tolist()
        self.max_iter = max_iter
        self.counter = 0
        
    def step(self):
        w = self.decays[self.counter]
        self.counter += 1
        self.counter = min(self.counter, self.max_iter-1)
        return w
        
        
def masked_reconstruction_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    return (loss * mask).sum() / mask.sum()

@torch.no_grad()
def ema_update(ema_model, model, buffers=True, decay=.999):
    for p_avg, p in zip(ema_model.parameters(), model.parameters()):
        p_avg.data = decay * p_avg.data + (1. - decay) * p.data
    if buffers:
        for (n, b_avg), (n2, b) in zip(ema_model.named_buffers(), model.named_buffers()):
            if n.split('.')[-1] == 'num_batches_tracked':
                b_avg.data = b.data
            else:
                b_avg.data = decay * b_avg.data + (1. - decay) * b.data

def train_step(student, teacher, optimizer, train_loader, scheduler=None, device='cuda', clip_norm=4., mask_ratio=0.8, ema_scheduler=None):
    losses = []
    
    student.train()
    teacher.eval()
    
    for x in tqdm(train_loader, leave=False):
        x = x.to(device)  # B=12, 1, T=512, F=128
    
        # with torch.no_grad():
        #     x = data_transforms(x)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            cls_pred, patch_pred, mask = student(x, mask_ratio)  # cls_pred: (B*clone_size, D), patch_pred: (B*clone_size, L=256, D), mask: (B*clone_size, L=256)
            with torch.no_grad():
                patch_target = teacher(x)  # (B*clone_size, L=256, D)
        
        cls_pred, patch_pred, mask, patch_target = cls_pred.float(), patch_pred.float(), mask.float(), patch_target.float()  
        cls_target = patch_target.mean(dim=1)  # B*clone_size, 768 
        
        # local (patch) loss + global (utterance) loss
        loss = masked_reconstruction_loss(patch_pred, patch_target, mask)
        loss += torch.mean((cls_pred - cls_target) ** 2.)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        
        if ema_scheduler is not None:
            decay = ema_scheduler.step()
        else:
            decay = 0.999
        ema_update(teacher.encoder, student.encoder, decay=decay)
        
        losses.append(loss.detach().cpu().item())
    
    return np.mean(losses)

# config
epochs = 30
device = 'cuda'
decoder_type = 'cnn2d'  # cnn2d, cnn1d, mlplstm, vit, swin
sample_dur = 5
patch_size = (16, 16)
clone_size = 16
mask_ratio = 0.8
clip_norm= 4.
# if decoder_type == 'cnn2d':
#     decoder_cls = 
#     decoder_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6}
mask_mode = 'inv'
use_mixup = False

experiment_name = f"EAT_{decoder_type}_{patch_size[0]}x{patch_size[1]}_{mask_mode}mask_{clone_size}clone"
if use_mixup:
     experiment_name += "_mixup"
print(experiment_name)

# saving and monitoring
par_dir = Path(f'logs/')
Path.mkdir(par_dir, parents=True, exist_ok=True)
time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
history_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_history.csv')
state_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_state.pt')
print(history_path)
print(state_path)
history = {'train_loss': []}

# build models
student = EAT_Student().to(device).train()
teacher = EAT_Teacher().to(device).eval()
teacher.encoder.load_state_dict(student.encoder.state_dict())
teacher.requires_grad_(False)
student.compile(mode='default')
teacher.compile(mode='default')
optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.05)  # betas=[0.9, 0.95]
scheduler = RiseRunDecay(optimizer, steps_in_epoch=len(train_loader), warmup=4, total_epochs=epochs, lowest_lr=1e-6)
print(f'#parameters: {sum(p.numel() for p in student.parameters()):_}')
print(f'#parameters: {sum(p.numel() for p in teacher.parameters()):_}')
ema_scheduler = EMA_Weight_Decay_Scheduler()

# run
if __name__ == "__main__":
  train_loader = ...
  pbar = tqdm(range(epochs), colour='orange')
  for epoch in pbar:
      train_loss = train_step(student, teacher, optimizer, train_loader, scheduler, device, clip_norm, mask_ratio, ema_scheduler)
      history['train_loss'].append(train_loss)
      if epoch in [0, 4, 9, 14, 19, 24, 29]:  # to make plots later for performance at different epochs
          torch.save({'encoder': student.encoder.state_dict(), 'ema_encoder': teacher.encoder.state_dict(), 'decoder': student.decoder.state_dict(), 'opt': optimizer.state_dict(), 'epochs': epochs}, state_path.as_posix().split('.pt')[0] + f"_epoch{epoch+1}.pt")
    pbar.set_description(f"loss={train_loss:.4f}")

  _ = pd.DataFrame(history).to_csv(history_path)
