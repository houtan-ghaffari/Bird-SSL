from pathlib import Path
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F


# Decoders:

## CNN-2d
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

## MLP-LSTM
class MLP_LSTM_Block(nn.Module):
    
    def __init__(self, dim=768, drop=0., activation=nn.GELU, bidirectional=True, add_residual=True):
        super().__init__()
        self.add_residual = add_residual
        self.act = activation()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=bidirectional)
        fc2_inp_size = int(dim * 2) if bidirectional else dim
        self.fc2 =  nn.Linear(fc2_inp_size, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
                
    def forward(self, x):
        self.lstm.flatten_parameters()
        z = self.fc1(x)
        z = self.act(z)
        z = self.ln1(z)
        z = self.drop(z)
        with torch.autocast(device_type='cuda', enabled=False):  # device_type=x.device
            z = z.float()
            z, _ = self.lstm(z)
        z = self.fc2(z)
        z = self.act(z)
        out = self.ln2(z)
        if self.add_residual:
            out = out + x
        return out

class MLP_LSTM_Decoder(nn.Module):
    def __init__(self, dim=768, drop=0., activation=nn.GELU, bidirectional=True, add_residual=True, num_layers=2, **kwargs):
        super().__init__()
        self.blocks = nn.Sequential(*[MLP_LSTM_Block(dim, drop, activation, bidirectional, add_residual) for _ in range(num_layers)])
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
    def forward(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore)
        return self.blocks(x)


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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
        

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, scale_norm=False, proj_bias=True, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        """
        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        # if qk_norm or scale_norm:
            # assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
      
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        

class AltBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_norm_first=False, ffn_targets=True):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rel_pos_bias=None, pos_mask=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            t = self.mlp(self.norm2(x))
            x = x + self.drop_path(t)
            if not self.ffn_targets:
                t = x
            return x, t
        else:
            x = x + self.drop_path(self.attn(x))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(x))
            if not self.ffn_targets:
                t = x
            return x, t
            

class PatchEmbed(nn.Module):

    def __init__(self, img_size=(512, 128), patch_size=(16, 16), in_chans=1, emb_dim=768):
        super().__init__()
        assert isinstance(patch_size, tuple)
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 256
        self.patch_ft = (img_size[1] // patch_size[1], img_size[0] // patch_size[0]) # number of patches height/width = 8/32
        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=patch_size, stride=patch_size)
       
    def forward(self, x):
        x = self.proj(x) # B, C=1, T=512, F=128 -> B, 768, 32, 8
        x = x.flatten(2) # B, 768, 32, 8 -> B, 768, 256
        x = x.transpose(1, 2) # B, 768, 256 -> B, 256, 768
        return x


# TODO: add sequential masking for using regular transformer (patch-size=(f=128, t=1))
class ViT_MaskedEncoder(nn.Module):
    def __init__(self, input_shape=(512, 128), patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path_rate=0., pos_trainable=False, clone_size=16, mode='student', mask_mode='inv'):
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
        pos_embed = get_2d_sincos_pos_embed_flexible(embed_dim, self.patch_embed.patch_ft, cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) # 1, 1, 768
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([AltBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
    def inverse_block_mask(self, shape, mask_ratio=0.8, num_freq_patches=8, num_time_patches=32, mask_length=5, mask_prob_adjust=0.07, require_same_masks=True):
        
        if mask_ratio == 0:
            return None, None, None
        
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
            return None, None, None
        
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
                
    def student_forward(self, x, mask_ratio=None):
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
            x, _ = blk(x)
        x = self.norm(x)

        # separate cls from the rest
        cls_pred = x[:, 0]
        patch_pred = x[:, 1:]
        
        return cls_pred, patch_pred, mask, ids_restore

    def teacher_forward(self, x, mask_ratio=None):
        x = self.patch_embed(x)  # B, C=1, T=512, F=128 -> B, L=256, D=768
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        features = []
        for blk in self.blocks:
            x, t = blk(x)
            features.append(t[:, 1:, :].clone())
        return features
       
    def forward(self, x, mask_ratio=None):
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
                 drop=0.,
                 attn_drop=0.,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=16,
                 mask_mode='inv',
                 decoder_cls=CNN2dDecoder,
                 decoder_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6},
                ):
        
        super().__init__()

        # student encoder & decoder
        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='student', mask_mode=mask_mode)
        decoder_kwargs['num_freq_patches'] = self.encoder.patch_embed.patch_ft[0]
        decoder_kwargs['num_time_patches'] = self.encoder.patch_embed.patch_ft[1]
        decoder_kwargs['dim'] = embed_dim
        self.decoder = decoder_cls(**decoder_kwargs)
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
               drop=0,
               attn_drop=0.,
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

        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='teacher')
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
        self.decays = np.linspace(decay_start, decay_end, max_iter, dtype=np.float32).tolist() + [decay_end]
        self.max_iter = max_iter
        self.counter = 0
        
    def step(self):
        w = self.decays[self.counter]
        self.counter = min(self.counter + 1, self.max_iter)
        return w


class Sigmoid_Rampup_Scheduler:
    def __init__(self, scale=-5.0, max_iter=None):
        self.scale = scale
        self.max_iter = max_iter
        self.counter = 0
        
    def step(self):
        phase_square = (1.0 - self.current / self.max_iter) ** 2
        self.counter = min(self.counter + 1, self.max_iter)
        return math.exp(self.scale * phase_square)
        
        
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
                

def load_eat_audioset_pretrained_state(model, audioset_eat_state_path='EAT-base_epoch30_pt.pt'):
    audioset_state = torch.load(eat_state_path)
    model_state = model.state_dict()
    model_state['cls_token'] = audioset_state['model']['modality_encoders.IMAGE.extra_tokens'].clone()
    model_state['patch_embed.proj.weight'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.weight'].clone()
    model_state['patch_embed.proj.bias'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.bias'].clone()
    # model_state['pos_embed'] = audioset_state['model']['modality_encoders.IMAGE.fixed_positional_encoder.positions'][:, :257].clone() not necessary
    model_state['norm.weight'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.weight'].clone()
    model_state['norm.bias'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.bias'].clone()
    for k in audioset_state['model'].keys():
        if k[:6] == 'blocks':
            model_state[k] = audioset_state['model'][k].clone()
    _ = model.load_state_dict(model_state, strict=False)
    print(_)
    return model


def train_step(student, teacher, optimizer, train_loader, scheduler=None, device='cuda', clip_norm=4., mask_ratio=0.8, ema_scheduler=None, teacher_assistant=None, sigmoid_scheduler=None):
    losses = []
    
    student.train()
    teacher.eval()
    
    for x in tqdm(train_loader, leave=False):
        x = x.to(device)  # B=12?, C=1, T=512, F=128
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            cls_pred, patch_pred, mask = student(x, mask_ratio)  # cls_pred: (B*clone_size, D=768), patch_pred: (B*clone_size, L=256, D), mask: (B*clone_size, L=256)
            with torch.no_grad():
                patch_target = teacher(x)  # (B*clone_size, L=256, D)
                if teacher_assistant is not None:
                    patch_target_assistant = teacher_assistant(x)  # (B*clone_size, L=256, D)
        
        cls_pred, patch_pred, mask, patch_target = cls_pred.float(), patch_pred.float(), mask.float(), patch_target.float()
        cls_target = patch_target.mean(dim=1)  # B*clone_size, D=768
        
        if teacher_assistant is not None:
            patch_target_assistant = patch_target_assistant.float()
            cls_target_assistant = patch_target_assistant.mean(dim=1)
            alpha = sigmoid_scheduler.step()  # alpha gradually goes from ~0 -> 1 with every step
            patch_target = alpha * patch_target + (1. - alpha) * patch_target_assistant
            cls_target = alpha * cls_target + (1. - alpha) * cls_target_assistant
            
        # local (patch) loss + global (utterance) loss
        loss = masked_reconstruction_loss(patch_pred, patch_target, mask)
        loss += torch.mean((cls_pred - cls_target) ** 2.)
        
        loss.backward()
        
        if clip_norm is not None:
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
seed = 42
sample_dur = 5
patch_size = (16, 16)
clone_size = 16
mask_mode = 'inv'  # rand, inv, seq 
mask_ratio = 0.8
clip_norm= 4.
encoder_type = 'vitbase'  # vitbase, vitlarge, vithuge
decoder_type = 'mlplstm'  # cnn2d, mlplstm
use_double_teacher = False

experiment_name = f"EAT_{encoder_type}_{decoder_type}_{patch_size[0]}x{patch_size[1]}_{mask_mode}mask{int(mask_ratio*100)}_{clone_size}clone"
if use_double_teacher:
     experiment_name += "_2teachers"
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

if decoder_type == 'cnn2d':
    decoder_cls = CNN2dDecoder
    decoder_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6}
elif decoder_type == 'mlplstm':
    decoder_cls = MLP_LSTM_Decoder
    decoder_kwargs = {'drop': 0, 'activation': nn.GELU, 'bidirectional': True, 'add_residual': True, 'num_layers': 2}
else:
    raise ValueError(f'{decoder_type} is not defined!')

if encoder_type == 'vitbase':
    embed_dim = 768
    depth = 12
    num_heads = 12
    average_top_k_layers = depth
    if use_double_teacher:
        audioset_eat_state_path = 'EAT-base_epoch30_pt.pt'
elif encoder_type == 'vitlarge':
    raise ValueError('later define the values for large and huge models here.')
elif encoder_type == 'vithuge':
    raise ValueError('later define the values for large and huge models here.')
else:
    raise ValueError('undefined model size.')


student = EAT_Student(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, clone_size=clone_size, mask_mode=mask_mode, emdecoder_cls=decoder_cls, decoder_kwargs=decoder_kwargs).to(device).train()
teacher = EAT_Teacher(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, clone_size=clone_size, average_top_k_layers=average_top_k_layers).to(device).eval()
teacher.encoder.load_state_dict(student.encoder.state_dict())
teacher.requires_grad_(False)
if use_double_teacher:
    teacher_assistant = EAT_Teacher(patch_size=(16, 16), embed_dim=embed_dim, depth=depth, num_heads=num_heads, clone_size=clone_size, average_top_k_layers=average_top_k_layers)
    teacher_assistant = load_eat_audioset_pretrained_state(teacher_assistant, audioset_eat_state_path=audioset_eat_state_path)
    teacher_assistant.to(device).eval()
    teacher_assistant.requires_grad_(False)
    sigmoid_scheduler = Sigmoid_Rampup_Scheduler(max_iter=epochs*len(train_loader)) 
else:
    teacher_assistant = None
    sigmoid_scheduler = None
    
optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.05, betas=[0.9, 0.95])
scheduler = RiseRunDecay(optimizer, steps_in_epoch=len(train_loader), warmup=epochs//8, total_epochs=epochs, lowest_lr=1e-6)
ema_scheduler = EMA_Weight_Decay_Scheduler(decay_start=0.9998, decay_end=0.99999, max_iter=len(train_loader)*(epochs//4))

print(f'#parameters: {sum(p.numel() for p in student.parameters()):_}')
print(f'#parameters: {sum(p.numel() for p in teacher.parameters()):_}')


# run
if __name__ == "__main__":
    train_loader = None
    pbar = tqdm(range(epochs), colour='orange')
    for epoch in pbar:
        train_loss = train_step(student, teacher, optimizer, train_loader, scheduler, device, clip_norm, mask_ratio, ema_scheduler, teacher_assistant, sigmoid_scheduler)
        history['train_loss'].append(train_loss)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save({'encoder': student.encoder.state_dict(), 'ema_encoder': teacher.encoder.state_dict(), 'decoder': student.decoder.state_dict(), 'opt': optimizer.state_dict(), 'epochs': epochs}, state_path.as_posix().split('.pt')[0] + f"_epoch{epoch+1}.pt")
        pbar.set_description(f"loss={train_loss:.4f}")
    _ = pd.DataFrame(history).to_csv(history_path)
