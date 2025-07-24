from pathlib import Path
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F


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
       
    def forward(self, x, mask_ratio=None):  #, mask_indices=None, no_mix_region_size=None): 
        return self.forward_fn(x, mask_ratio)
      

# Student model for SSLAM (same as EAT)
class SSLAM_Student(nn.Module):
    
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
                 clone_size=8,
                 mask_mode='inv',
                 decoder_cls=CNN2dDecoder,
                 decoder_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6},
                ):
        
        super().__init__()

        # student encoder & decoder
        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='student', mask_mode=mask_mode)
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
    
    def forward(self, x, mask_ratio=None):
        """
        args:
            x - input mel-spectrogram of shape B, 1, T, F 
        """
        cls_pred, patch_pred, mask, ids_restore = self.encoder(x, mask_ratio)
        patch_pred = self.decoder(patch_pred, ids_restore)
        return cls_pred, patch_pred, mask
      

# Teacher model for SSLAM
class SSLAM_Teacher(nn.Module):
    
    def __init__(self,
                 input_shape=(512, 128), 
                 patch_size=(16, 16),
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop=0,
                 attn_drop=0,
                 drop_path_rate=0,
                 pos_trainable=False,
                 clone_size=8,
                 average_top_k_layers=12,
                 instance_norm_target_layer=True,  # based on EAT config
                 batch_norm_target_layer=False,  # based on EAT config
                 layer_norm_target_layer=False,  # based on EAT config
                 layer_norm_targets=True,  # based on EAT config
                 instance_norm_targets=False,  # based on EAT config
                ):
        
        super().__init__()

        self.encoder = ViT_MaskedEncoder(input_shape, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path_rate, pos_trainable, clone_size, mode='teacher')
        # self.num_freq_patches, self.num_time_patches = self.encoder.patch_embed.patch_ft
        self.clone_size = clone_size
        self.average_top_k_layers = average_top_k_layers
        self.instance_norm_target_layer = instance_norm_target_layer
        self.batch_norm_target_layer = batch_norm_target_layer
        self.layer_norm_target_layer = layer_norm_target_layer
        self.layer_norm_targets = layer_norm_targets
        self.instance_norm_targets = instance_norm_targets

    def prepare_dropped_inputs_for_teacher_after_pos(self, patch_inp, mix_indices, no_mix_region_size=112):
        """
        Args:
            patch_inp (Tensor): INPUT after patch embedding. (batch_size, seq_len+1(cls), emb_dim)
            mask_indices (List): The indices where mixing is not done. all are multiples of 16

        Returns:
            Tensor: mixed_parts (Tensor): The input tensor with the mixed region.

        AUDIO 1: =============idx1 XXXXXXXXXXXXXX idx1+no_mix_region_size=============idx2 XXXXXXXXXXXXXX idx2+no_mix_region_size=============
                  part1                                                        part2                                                part3
        RETURN : CONCAT(part1, part2, part3)
        """
        batch_size, _, emb_dim = patch_inp.shape
        patch_size = self.encoder.patch_size[0]
        num_freq_patches, num_time_patches = self.encoder.patch_embed.patch_ft

        # restore 2d patch order (batch_size,768,T,F) 
        cls_token = patch_inp[:, 0, :].unsqueeze(1)  # (batch_size, 1, 768)
        patch_inp_without_cls = patch_inp[:, 1:, :]  # remove cls token
        patch_inp_without_cls = patch_inp_without_cls.permute(0, 2, 1)  # (batch_size, 768, 256)
        patch_inp_without_cls = patch_inp_without_cls.reshape(batch_size, emb_dim, num_time_patches, num_freq_patches) ## (batch_size, 768, 32, 8)

        # convert mix_indices to patch indices by dividing by patch_size(16)
        idx1, idx2, no_mix_region_size = mix_indices[0] // patch_size, mix_indices[1] // patch_size, no_mix_region_size // patch_size

        # take mixed parts
        part1 = patch_inp_without_cls[:, :, :idx1, :]
        part2 = patch_inp_without_cls[:, :, idx1+no_mix_region_size:idx2, :]
        part3 = patch_inp_without_cls[:, :, idx2+no_mix_region_size:, :]
        mixed_parts = torch.cat((part1, part2, part3), dim=-2)
        
        # restore flattened patch order (batch_size, 144, 768)
        mixed_parts = mixed_parts.reshape(batch_size, 768, -1)  # (batch_size, 768, 144)
        mixed_parts = mixed_parts.permute(0, 2, 1)  # (batch_size, 144, 768)
        assert mixed_parts.shape[1:] == (144, 768), f"unexpected mixed_parts shape: {mixed_parts.shape}"

        # attach cls token
        mixed_parts = torch.cat((cls_token, mixed_parts), dim=1) ## (batch_size, 145, 768)
        return mixed_parts

    def get_patch_target_for_mix(self, patch_latents_source, patch_latents_source_mixed_parts, num_layers, mix_indices, no_mix_region_size):
        """
        AUDIO 2(patch_latents_source): ============================================

        AUDIO 1(patch_latents_mix): CONCAT(======XdroppedX=======XdroppedX========)
                                            part1          part2           part3
        Args:
            patch_latents_source: all layers' output for non mixed half of batch #[(,256,768)......]
            patch_latents_mix: all layers' output from SECOND Forward with source after dropping non-mixed parts #[(,145,768)......]
            mix_indices: indices of rolled batch where we didnt mix it and kept the source in those regions
            no_mix_region_size: length of the intervals of not mixing
        """
        
        y_patch_sslam_source = self.make_targets(patch_latents_source, num_layers)  # (B, L=256, D=768)
        y_patch_sslam_source_mixed_parts = self.make_targets(patch_latents_source_mixed_parts, num_layers)  # (B, L=144, D=768)
        y_patch_sslam_roll_mixed_parts = torch.roll(y_patch_sslam_source_mixed_parts, shifts=1, dims=0)  # the second batch for mixing was created by rolling

        patch_size = self.encoder.patch_size[0]  # 16
        num_freq_patches, num_time_patches = self.encoder.patch_embed.patch_ft  # 8, 32
        batch_size, _, emb_dim = y_patch_sslam_source.shape

        # the unmixing parts of target will come from source and mixing parts from the avg of the source and its rolled version
        idx1, idx2, no_mix_region_size = mix_indices[0] // patch_size, mix_indices[1] // patch_size, no_mix_region_size // patch_size
        idx2_skipped = idx2 - no_mix_region_size  # there is no dropping here; just skipping the dropped (unmixed) patches and setting them to source values 
        
        y_reconstructed = y_patch_sslam_source.clone().permute(0, 2, 1)  # (B, D=768, L=256)
        y_reconstructed = y_reconstructed.reshape(batch_size, emb_dim, num_time_patches, num_freq_patches) ## (B, D=768, t=32, f=8)
        y_patch_sslam_roll_mixed_parts = y_patch_sslam_roll_mixed_parts.permute(0, 2, 1)  # (B, 768, 144)
        y_patch_sslam_roll_mixed_parts = y_patch_sslam_roll_mixed_parts.reshape(batch_size, emb_dim, -1, num_freq_patches)  # (B, D=768, t=18, f=8)
        
        y_reconstructed[:, :, :idx1, :] = y_patch_sslam_roll_mixed_parts[:, :, :idx1, :]  # part 1
        y_reconstructed[:, :, idx1+no_mix_region_size:idx2, :] = y_patch_sslam_roll_mixed_parts[:, :, idx1:idx2_skipped, :]  # part2
        y_reconstructed[:, :, idx2+no_mix_region_size:, :] = y_patch_sslam_roll_mixed_parts[:, :, idx2_skipped:, :]  # part3
        
        y_reconstructed = y_reconstructed.reshape(batch_size, emb_dim, -1).permute(0, 2, 1)  # (B, L=256, D=768)
        
        y_patch_sslam = (y_patch_sslam_source + y_reconstructed) / 2.0  # (B, L, D)
        return y_patch_sslam

    def get_cls_target_for_mix(self, patch_latents_source, patch_latents_source_mixed_parts):
        # y_cls_sslam_source = [t_i.clone() for t_i in patch_latents_source]
        # y_cls_sslam_mix = [t_i.clone() for t_i in patch_latents_mix]  # safety from inplace operation
        y_cls_sslam_source = self.make_targets(patch_latents_source, 1)
        y_cls_sslam_source_mixed_parts = self.make_targets(patch_latents_source_mixed_parts, 1)
        y_cls_sslam_roll_mixed_parts = torch.roll(y_cls_sslam_source_mixed_parts, shifts=1, dims=0)

        # do these in float32 later
        # y_cls_sslam_source_1 = y_cls_sslam_source_1.mean(dim=1)  # (b, 768)
        # y_cls_sslam_source_2 = y_cls_sslam_source_2.mean(dim=1)  # (b, 768)
        
        y_cls_sslam = (y_cls_sslam_source + y_cls_sslam_roll_mixed_parts) / 2.0  # (b, 256, 768)
        return y_cls_sslam

    def make_targets(self, layers_outputs, layers_to_avg_from_last):
        
        y = layers_outputs[-layers_to_avg_from_last:]
        y = [out.clone() for out in y]  # safety from inplace operation
        
        permuted = False
        if self.instance_norm_target_layer or self.batch_norm_target_layer:
            y = [tl.transpose(1, 2) for tl in y]  # BTC -> BCT
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

    def forward_mixed_parts(self, x, mix_indices, no_mix_region_size):
        z = self.encoder.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        cls_tokens = self.encoder.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat((cls_tokens, z), dim=1)
        z = z + self.encoder.pos_embed
        z = self.prepare_dropped_inputs_for_teacher_after_pos(z, mix_indices, no_mix_region_size)
        patch_latents_mixed_parts = []
        for blk in self.encoder.blocks:
            z, t = blk(z)
            patch_latents_mixed_parts.append(t[:, 1:, :].clone())
        return patch_latents_mixed_parts
        
    def forward(self, x, mix_indices, no_mix_region_size):
        """
        args:
            x - input mel-spectrogram of source and mixed: 2*B, 1, T, F 
        """
        x_source = x.chunk(2, dim=0)[0].clone()

        # embbeding source and mix of source with its rolled version
        patch_latents = self.encoder(x, 0)  # outputs a list of all transformer layers' embeddings (e.g., 12 layer in base vit)
        patch_latents_source = [t_i.chunk(2, dim=0)[0].clone() for t_i in patch_latents]
        
        # embbeding source after dropping unmixed parts
        patch_latents_source_mixed_parts = self.forward_mixed_parts(x_source, mix_indices, no_mix_region_size)
        
        # for EAT
        y_patch_eat = self.make_targets(patch_latents, self.average_top_k_layers)  # source and mixed 2*B, L, D
        y_cls_eat = self.make_targets(patch_latents, 1)  # source and mixed 2*B, L, D; avg in dim=1 in float32 later

        # for SSLAM
        y_patch_sslam = self.get_patch_target_for_mix(patch_latents_source, patch_latents_source_mixed_parts, self.average_top_k_layers, mix_indices, no_mix_region_size)
        # calculated but not used in their code or paper.
        # y_cls_sslam = self.get_cls_target_for_mix(patch_latents_source, patch_latents_mixed_parts)
        y_cls_sslam = None
        
        return y_patch_eat, y_cls_eat, y_patch_sslam, y_cls_sslam


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
        self.counter += 1
        self.counter = min(self.counter, self.max_iter)
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
        x = x.to(device)  # B, C=1, T=512, F=128
        
        x_roll = torch.roll(x, shifts=1, dims=0)  # make a second batch for mixing two audio by rolling on batch axis
        
        # indices where to not mix (multiples of 16 because of 16x16 patch_size)
        no_mix_region_size = 112
        idx_1 = random.choice([i for i in range(0, 251, 16)]) # random multiple of 16 between 0 and 250
        idx_2 = random.choice([i for i in range(idx_1+no_mix_region_size, 511-no_mix_region_size, 16)])
        mix_indices = [idx_1, idx_2]
        
        # drop the regions between indices
        for idx in mix_indices:
            x_roll[:, :, idx:idx+no_mix_region_size, :] = float('-inf') ## no mixing in that region, original spect will be used
        
        # mix
        x_mixed = torch.max(x, x_roll)
        
        # concat unmixed and mixed (2*b, 1, 512, 128)
        x = torch.cat((x, x_mixed), dim=0)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # student only processes masked versions of unmixed and mixed inputs
            cls_pred, patch_pred, mask = student(x, mask_ratio)
            # teacher produces latent targets for:
            # 1) the original and mixed input (x) to use for EAT loss
            # 2) average of the mixed parts of the rolled and original input to use for SSLAM loss
            with torch.no_grad():
                y_patch_eat, y_cls_eat, y_patch_sslam, y_cls_sslam = teacher(x, mix_indices, no_mix_region_size)
        
        cls_pred = cls_pred.float()  # 2*B*clone, D=768
        patch_pred = patch_pred.float()  # 2*B*clone, L=256, D=768
        mask = mask.float()  # 2*B*clone, L=256
        y_patch_eat = y_patch_eat.float().repeat_interleave(teacher.clone_size, 0)  # 2B*clone, L=256, D=768
        y_patch_sslam = y_patch_sslam.float().repeat_interleave(teacher.clone_size, 0)  # B*clone, L=256, D=768
        y_cls_eat = y_cls_eat.float().mean(dim=1).repeat_interleave(teacher.clone_size, 0) # 2B*clone, D=768
        # y_cls_sslam = y_cls_sslam.float().mean(dim=1).repeat_interleave(teacher.clone_size, 0)  # B*clone, D  not used apparently
        
        # EAT loss
        eat_cls_loss = torch.mean((cls_pred - y_cls_eat) ** 2.)            
        eat_patch_loss = masked_reconstruction_loss(patch_pred, y_patch_eat, mask)

        # SSLAM mix loss
        _, patch_pred_mixed = patch_pred.chunk(2, dim=0)
        _, mask_mixed = mask.chunk(2, dim=0)
        sslam_patch_loss = masked_reconstruction_loss(patch_pred_mixed, y_patch_sslam, mask_mixed)
       
        loss = eat_cls_loss + eat_patch_loss + sslam_patch_loss * 2.0

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
seed = 42
sample_dur = 5
patch_size = (16, 16)
clone_size = 16
mask_mode = 'inv'
mask_ratio = 0.8
clip_norm= 4.
decoder_type = 'cnn2d'  # cnn2d, cnn1d, mlplstm, vit, swin
# if decoder_type == 'cnn2d': for ablation, later...
#     decoder_cls = LSTM or CNN2D
#     decoder_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same', 'groups': 16, 'activation': nn.GELU, 'add_residual': True, 'num_layers': 6}

experiment_name = f"SSLAM_{decoder_type}_{patch_size[0]}x{patch_size[1]}_{mask_mode}mask{int(mask_ratio*100)}_{clone_size}clone"
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
student = SSLAM_Student().to(device).train()
teacher = SSLAM_Teacher().to(device).eval()
teacher.encoder.load_state_dict(student.encoder.state_dict())
teacher.requires_grad_(False)
# student.compile(mode='default') do not compile the models for now. There are issues with indexing that makes it to recompile a lot and waste time.
# teacher.compile(mode='default')
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
        train_loss = train_step(student, teacher, optimizer, train_loader, scheduler, device, clip_norm, mask_ratio, ema_scheduler)
        history['train_loss'].append(train_loss)
        if epoch in [0, 4, 9, 14, 19, 24, 29]:  # to make plots later for performance at different epochs
            torch.save({'encoder': student.encoder.state_dict(), 'ema_encoder': teacher.encoder.state_dict(), 'decoder': student.decoder.state_dict(), 'opt': optimizer.state_dict(), 'epochs': epochs}, state_path.as_posix().split('.pt')[0] + f"_epoch{epoch+1}.pt")
        pbar.set_description(f"loss={train_loss:.4f}")

    _ = pd.DataFrame(history).to_csv(history_path)
