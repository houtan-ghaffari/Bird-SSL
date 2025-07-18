from pathlib import Path
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import get_2d_sincos_pos_embed_flexible


# Vit Encoder for original EAT with AudioSet pretraining
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


class ViT(nn.Module):
    def __init__(self, num_classes=None, input_shape=(512, 128), patch_size=(16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path_rate=0., pos_trainable=False, mask_mode='rand'):
        super().__init__()
        assert mask_mode in ['rand', 'inv']
        if mask_mode == 'rand':
            self.mask_fn = self.random_masking
        else:
            self.mask_fn = self.inverse_block_mask
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_shape, patch_size, 1, embed_dim)
        pos_embed = get_2d_sincos_pos_embed_flexible(embed_dim, self.patch_embed.patch_ft, cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=pos_trainable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) # 1, 1, 768
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([AltBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    # def resize_pos_emedding(self, new_size=5000):
        # embed_dim = self.pos_embed.shape[-1]
        # self.register_buffer("pos_embed", get_positional_encoding(embed_dim, max_len=new_size))
        
    def inverse_block_mask(self, shape, mask_ratio=0.2, num_freq_patches=8, num_time_patches=32, mask_length=5, mask_prob_adjust=0.07, require_same_masks=True):
        
        if mask_ratio == 0:
            return None, None, None
        
        assert mask_length > 1
        
        B, L, D = shape
        # B = B * self.clone_size
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
    
    def random_masking(self, shape, mask_ratio, *args):
        
        if mask_ratio == 0:
            return None, None, None
        
        B, L, D = shape  # batch, length, dim
        # B *= self.clone_size
        
        len_keep = int(L * (1 - mask_ratio))    
        noise = torch.rand(B, L)  # noise in [0, 1]
        
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
    
    def forward(self, x, mask_ratio=0.2):
        x = self.patch_embed(x)  # B, 1, T, F -> B, L=256, D=768
        x = x + self.pos_embed[:, 1:, :]
        num_freq_patches, num_time_patches = self.patch_embed.patch_ft
        mask, ids_keep, ids_restore = self.mask_fn(x.shape, mask_ratio, num_freq_patches, num_time_patches)
        if mask is not None:
            x = torch.gather(x, dim=1, index=ids_keep.to(x.device))  # B, L * (1 - mask_ratio), D

        # add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # encode
        for blk in self.blocks:
            x, t = blk(x)
        x = self.norm(x)

        # predict
        cls_tokens = x[:, 0]
        return self.fc(cls_tokens)


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


def train_step(model, optimizer, train_loader, ema_model=None, scheduler=None, device='cuda', clip_norm=4., mask_ratio=0.2, ema_scheduler=None):
    losses = []
    model.train()
    for x, y in tqdm(train_loader, leave=False):
        x, y = x.to(device), y.to(device)  # x: B, 1, T=512, F=128 ; y: B, C=num_classes
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x, mask_ratio)
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().item())
        if ema_model is not None:
            if ema_scheduler is not None:
                decay = ema_scheduler.step()
            else:
                decay = 0.999
            ema_update(ema_model, model, decay=decay)
    return np.mean(losses)


@torch.inference_mode()
def val_step(model_, val_loader, device='cuda', prefix='val_'):
    y_true, y_pred, loss = [], [], []
    model_.eval()
    for x, y in tqdm(val_loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model_(x, 0)
        logits = logits.float()
        loss.append(F.binary_cross_entropy_with_logits(logits, y).cpu().item())
        y_true.append(y.cpu())
        y_pred.append(logits.sigmoid().cpu())
    cache = get_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), prefix)
    cache[prefix+'loss'] = np.mean(loss)
    return cache


def get_metrics(y_true, y_pred, prefix=''):
    
    y_pred_hard = y_pred.round()
    p, r, f, _ = metrics.precision_recall_fscore_support(y_true, y_pred_hard, average='macro', zero_division=0)
    cache = {prefix+'f1_macro': f * 100,
             prefix+'precision': p * 100,
             prefix+'recall': r * 100,
             prefix+'f1_micro': metrics.f1_score(y_true, y_pred_hard, average='micro', zero_division=0) * 100,
             prefix+'auc': metrics.roc_auc_score(y_true, y_pred, average='macro') * 100,
             prefix+'mAP': metrics.average_precision_score(y_true, y_pred, average='macro') * 100,
             }
    post_pos, post_neg = get_avg_posterior(y_true, y_pred)
    cache[prefix+'post_pos_mu'] = post_pos['mu'] * 100
    cache[prefix+'post_pos_std'] = post_pos['std'] * 100
    cache[prefix+'post_neg_mu'] = post_neg['mu'] * 100
    cache[prefix+'post_neg_std'] = post_neg['std'] * 100

    # TODO (@Lukas):  please check if the top1 accuracy is correct here. 
    # top1 accuracy: if the top predicted class is within the true labels
    y_true, y_pred = torch.from_numpy(y_true), torch.from_numpy(y_pred)
    mask = y_true.sum(dim=1) != 0
    # mask_no_call = ~mask
    # y_true_no_call = y_true[mask_no_call]
    # y_pred_no_call = y_pred[mask_no_call]  
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    top1_index = y_pred.argmax(dim=1)
    cache[prefix+'T1A'] = ((y_true[torch.arange(y_true.shape[0]), top1_index] == 1).float().sum() / y_true.shape[0]).item() * 100
    return cache


def get_avg_posterior(y_true, y_pred):
    post_mus_pos, post_mus_neg = {}, {}
    num_classes = y_pred.shape[1]
    for k in range(num_classes):
        y_true_k = y_true[:, k]
        y_pred_k = y_pred[:, k]
        post_mus_pos[k] = y_pred_k[y_true_k == 1].mean()
        post_mus_neg[k] = 1 - y_pred_k[y_true_k == 0].mean()
    post_mus_pos['mu'] = np.mean(list(post_mus_pos.values()))
    post_mus_pos['std'] = np.std(list(post_mus_pos.values()))
    post_mus_neg['mu'] = np.mean(list(post_mus_neg.values()))
    post_mus_neg['std'] = np.std(list(post_mus_neg.values()))
    return post_mus_pos, post_mus_neg

def load_eat_audioset_pretrained_state(model, eat_state_path='EAT-base_epoch30_pt.pt'):
    audioset_state = torch.load(eat_state_path)
    model_state = model.state_dict()
    model_state['cls_token'] = audioset_state['model']['modality_encoders.IMAGE.extra_tokens'].clone()
    model_state['patch_embed.proj.weight'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.weight'].clone()
    model_state['patch_embed.proj.bias'] = audioset_state['model']['modality_encoders.IMAGE.local_encoder.proj.bias'].clone()
    model_state['pos_embed'] = audioset_state['model']['modality_encoders.IMAGE.fixed_positional_encoder.positions'][:, :257].clone()
    model_state['norm.weight'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.weight'].clone()
    model_state['norm.bias'] = audioset_state['model']['modality_encoders.IMAGE.context_encoder.norm.bias'].clone()
    for k in audioset_state['model'].keys():
        if k[:6] == 'blocks':
            model_state[k] = audioset_state['model'][k].clone()
    _ = model.load_state_dict(model_state, strict=False)
    print(_)
    return model


def plot_history(history):
    _ = plt.figure(figsize=(21, 8))

    plt.subplot(251)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['ema_loss'], label='ema_loss')
    plt.plot(history['test_loss'], label='test_loss')
    plt.legend()

    plt.subplot(252)
    plt.plot(history['ema_f1_macro'], label='ema_f1_macro')
    plt.plot(history['test_f1_macro'], label='test_f1_macro')
    plt.legend()

    plt.subplot(253)
    plt.plot(history['ema_f1_micro'], label='ema_f1_micro')
    plt.plot(history['test_f1_micro'], label='test_f1_micro')
    plt.legend()

    plt.subplot(254)
    plt.plot(history['ema_precision'], label='ema_precision')
    plt.plot(history['test_precision'], label='test_precision')
    plt.legend()

    plt.subplot(255)
    plt.plot(history['ema_recall'], label='ema_recall')
    plt.plot(history['test_recall'], label='test_recall')
    plt.legend()

    plt.subplot(256)
    plt.plot(history['ema_T1A'], label='ema_T1A')
    plt.plot(history['test_T1A'], label='test_T1A')
    plt.legend()

    plt.subplot(257)
    plt.plot(history['ema_mAP'], label='ema_mAP')
    plt.plot(history['test_mAP'], label='test_mAP')
    plt.legend()

    plt.subplot(258)
    plt.plot(history['ema_auc'], label='ema_auc')
    plt.plot(history['test_auc'], label='test_auc')
    plt.legend()

    plt.subplot(259)
    mu = np.array(history['ema_post_pos_mu'])
    s = np.array(history['ema_post_pos_std'])
    plt.plot(mu, label='ema post pos $\mu$')
    plt.fill_between(range(mu.shape[0]),mu+s, mu-s, alpha=.5, label='ema post pos $\sigma$')
    mu = np.array(history['test_post_pos_mu'])
    s = np.array(history['test_post_pos_std'])
    plt.plot(mu, label='test post pos $\mu$')
    plt.fill_between(range(mu.shape[0]),mu+s, mu-s, alpha=.5, label='test post pos $\sigma$')
    plt.legend()

    plt.subplot(2,5,10)
    mu = np.array(history['ema_post_neg_mu'])
    s = np.array(history['ema_post_neg_std'])
    plt.plot(mu, label='ema post neg $\mu$ ')
    plt.fill_between(range(mu.shape[0]),mu+s, mu-s, alpha=.5, label='ema post neg $\sigma$')
    mu = np.array(history['test_post_neg_mu'])
    s = np.array(history['test_post_neg_std'])
    plt.plot(mu, label='test post neg $\mu$ ')
    plt.fill_between(range(mu.shape[0]),mu+s, mu-s, alpha=.5, label='test post neg $\sigma$')
    plt.legend()

    plt.show()


# config
epochs = 200
device = 'cuda'
model_type = 'vit_cnn2d'
sample_dur = 5.11
patch_size = (16, 16)
use_ema_pretrained = False
mask_mode = 'rand'
mask_ratio = 0.2
clip_norm = 4.
compression = None  # pcen or other variants; I can add the code later if you want
use_secondary_labels = False  # this helps me so much, doubling the metrics or even more sometimes!
use_tf_mask = False
use_bernoulli_noise = False  # this is better than tf-mask of SpecAugment in my experiments
use_striped_mask = False  # not good
use_patch_mask = False  # not good, this is similar to SSAST model masking in its pretraining. also, a bit similar to inverse-block masking, but quite the same.
use_pitch_shift = False  # the squeeze here doesn't worth the juice!
use_time_stretch = False  # not worth the troubles either, usually harmful in my experiments.
use_bg_noise = True  # VOX no-call dataset
use_color_noise = False
use_random_gain = False
use_mixup = True  # in my code, this mixup uses class-frequency weighted sampling, which helps a lot with unbalanced dataset.
double_mixup = False  # a second mixup at the batch level, which helped in my code
use_hrps = False  # harmonic-residual-percussive source separation and mixing again with random weights
# use_pie = False  this is for later \(^_^)/

experiment_name = f"AudioSet_{model_type}_{mask_mode}({int(100*mask_ratio)})_{patch_size[0]}x{patch_size[1]}"
if compression is not None: experiment_name += f"_{compression}"
if use_tf_mask: experiment_name += "_tfmask"
if use_bernoulli_noise: experiment_name += "_bern"
if use_striped_mask: experiment_name += "_striped"
if use_patch_mask: experiment_name += '_patchmask'
if use_pitch_shift: experiment_name += "_pitch"
if use_time_stretch: experiment_name += "_tstretch"
if use_bg_noise: experiment_name += "_bgvox"
if use_color_noise: experiment_name += "_colornoise"
if use_random_gain: experiment_name += "_gain"
if use_mixup: experiment_name += "_mixup"
if double_mixup: experiment_name += "2"
if use_hrps: experiment_name += "_hrps"
if not use_secondary_labels: experiment_name += "_singlelabel"
print(experiment_name)

# saving and monitoring
par_dir = Path(f'logs/')
Path.mkdir(par_dir, parents=True, exist_ok=True)
time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
history_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_history.csv')
state_path = par_dir.joinpath(f'{experiment_name}_{int(sample_dur)}sec_{time_id}_state.pt')
print(history_path)
print(state_path)
history = {'train_loss': [],
           'ema_loss': [],
           'ema_f1_macro': [],
           'ema_precision': [],
           'ema_recall': [],
           'ema_f1_micro': [],
           'ema_auc': [],
           'ema_mAP': [],
           'ema_post_pos_mu': [],
           'ema_post_pos_std': [],
           'ema_post_neg_mu': [],
           'ema_post_neg_std': [],
           'ema_T1A':[],
           'test_loss': [],
           'test_f1_macro': [],
           'test_precision': [],
           'test_recall': [],
           'test_f1_micro': [],
           'test_auc': [],
           'test_mAP': [],
           'test_post_pos_mu': [],
           'test_post_pos_std': [],
           'test_post_neg_mu': [],
           'test_post_neg_std': [],
           'test_T1A':[]}

# build models
model = ViT(num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, dropout=0.1, drop_path_rate=0, mask_mode=mask_mode)
ema_model = ViT(num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, dropout=0.1, drop_path_rate=0, mask_mode=mask_mode)
model = load_eat_audioset_pretrained_state(model, eat_state_path='EAT-base_epoch30_pt.pt')
ema_model.load_state_dict(model.state_dict())
model.to(device).train()
ema_model.to(device).eval()
model.compile(mode='default')  # reduce-overhead
ema_model.compile(mode='default')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05, betas=[0.9, 0.95])
scheduler = RiseRunDecay(optimizer, steps_in_epoch=len(train_loader), warmup=20, total_epochs=epochs)
ema_scheduler = EMA_Weight_Decay_Scheduler(decay_start=0.9998, decay_end=0.99999, max_iter=len(train_loader)*(epochs//4))
print(f'#parameters: {sum(p.numel() for p in model.parameters()):_}')


# run
if __name__ == "__main__":
    train_loader = None
    pbar = tqdm(range(epochs), colour='green')
for epoch in pbar:
    train_loss = train_step(model, optimizer, train_loader, ema_model, scheduler, device, clip_norm, mask_ratio, ema_scheduler)
    test_cache = val_step(model, test_loader, device, prefix='test_')
    ema_test_cache = val_step(ema_model, test_loader, data_transforms, prefix='ema_')
    history['train_loss'].append(train_loss)
    # if val_loader is not None:  We don't have validation for now...
    #     val_cache = val_step(model, val_loader, device, prefix='val_')
    #     for k, v in val_cache.items():
    #         history[k].append(v)
    #     val_loss = val_cache['val_loss']
    # else:
    #     val_loss = -1
    for k, v in test_cache.items():
        history[k].append(v)
    for k, v in ema_test_cache.items():
        history[k].append(v)
    pbar.set_description(f"loss={train_loss:.4f} test_loss={test_cache['test_loss']:.4f} test_mAP={test_cache['test_mAP']:.4f} test_f1={test_cache['test_f1_macro']:.4f}  ema_mAP={ema_test_cache['ema_mAP']:.4f} ema_f1={ema_test_cache['ema_f1_macro']:.4f}")
