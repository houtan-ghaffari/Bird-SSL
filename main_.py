#%%
import torch 
from datasets import load_dataset, load_from_disk
import os
import torchaudio
from datasets import Audio
import numpy as np 
from transformers import AutoFeatureExtractor
from torchaudio.compliance.kaldi import fbank

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data",
    split="train",
)
#%%

dataset[0]
#%%
from IPython.display import Audio as IPythonAudio
IPythonAudio(dataset[100]["audio"]["array"],
             rate=44_100)
#%%
# augmentations
# rolling start of a audio file (added at the end ot itself)
def cyclic_rolling_start(waveform):
    idx = np.random.randint(0, len(waveform))
    # rolls the np.array (waveform) to the right by idx samples
    rolled_waveform = np.roll(waveform, idx)
    # random volume magnitude
    volume_mag = np.random.beta(10,10) + 0.5

    waveform = rolled_waveform * volume_mag
    # normalize
    waveform = waveform - waveform.mean()
    return waveform


#%%

from transformers.audio_utils import spectrogram
from transformers.audio_utils import window_function
from transformers.audio_utils import mel_filter_bank

    # fbank_params = {
    #     'htk_compat': True,
    #     'sample_frequency': 44_100,
    #     'use_energy': False,
    #     'window_type': 'hanning',
    #     'num_mel_bins': 128,
    #     'dither': 0.0,
    #     'frame_shift': 10,
    # }

def transform(batch):
    sampling_rate = 44_100

    mel_filters = mel_filter_bank(
        num_frequency_bins=256,
        num_mel_filters=128, #bins
        min_frequency=20,
        max_frequency=sampling_rate//2,
        sampling_rate=sampling_rate,
        norm=None,
        mel_scale="kaldi",
        triangularize_in_mel_space=True
    )

    mel_filters = np.pad(mel_filters, ((0,1),(0,0)))
    window = window_function(400, "hann", periodic=False)

    spectrogram_params = {
        "window": window,
        "frame_length": 400,
        "hop_length": 160,
        "fft_length": 512,
        "power": 2.0,
        "center": False,
        "preemphasis":0.97,
        "mel_filters": mel_filters,
        "log_mel": "log",
        "mel_floor": 1.192092955078125e-07,
        "remove_dc_offset": True}


    input = []
    for audio in batch["audio"]:
        wav = cyclic_rolling_start(audio["array"])
        wav = spectrogram(wav, **spectrogram_params).T
        wav = torch.from_numpy(wav)
        #wav = fbank(wav.unsqueeze(0), **fbank_params)


        target_length = 512
        n_frames = wav.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            wav = m(wav)
        elif p < 0:
            wav = wav[0:target_length, :]
        
        #wav = wav.transpose(0,1)
        # wav = torch.transpose(wav.squeeze(), 0, 1)

        input.append(wav.unsqueeze(0))


    return {
        'audio': input, # mono: (1 channel x widht x height)
        'label': torch.Tensor(batch["target"]),
    }
dataset.set_transform(transform, output_all_columns=False)
#%%
dataset[100]["audio"].shape
#IPythonAudio(dataset[0]["audio"],rate=44_100)
#%%
import matplotlib.pyplot as plt
plt.imshow(dataset[100]["audio"].squeeze(0).T)

#%%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
next(iter(dataloader))


#%%
from functools import partial
from models import MaskedAutoencoderViT
import torch.nn as nn 
from timm.models.vision_transformer import PatchEmbed

target_length = 512 
in_chans = 1
img_size = (target_length, 128) # 512, 128

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        use_custom_patch=False, 
        in_chans=in_chans, img_size=img_size, audio_exp=True, norm_pix_loss=True, **kwargs) 
    #audio_exp important since audio not image
    # channles on 1 because mono 
    # image size depends on spectrogram 
    # norm_pixel loss for training stability
    return model

model = mae_vit_base_patch16()

# resize images
#model.patch_embed = PatchEmbed(img_size, 16, in_chans, 768)
#%%
#sample = dataset[0]["audio"]

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model.train(True)
#%%
for batch in dataloader:
    loss,_,_,_ = model(batch["audio"], mask_ratio=0.8)
    print(loss)
    break
#%%
datalaoder = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
next(iter(datalaoder))["audio"].shape

#%%
from tqdm import tqdm
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
device = "cuda"

def train_one_epoch(model, dataloader, optimizer, device, epoch=None):
    model.train(True)
    model.to(device)
    optimizer.zero_grad()

    model.epoch = epoch
    for batch in tqdm(dataloader):
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        loss,_,_,_ = model(audio, mask_ratio=0.8)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("loss: ", loss.item())
        #loss.backward()

    # Close the progress bar
    progress_bar.close()
#%%
train_one_epoch(model=mae_vit_base_patch16(), dataloader=dataloader, optimizer=optimizer, device="cuda")
#%%

from models import MAE_Sound
from tqdm import tqdm
import torch.nn as nn
from functools import partial

target_length = 512 
in_chans = 1
img_size = (target_length, 128) # 512, 128

model = MAE_Sound(
    patch_size=16, embed_dim=768, depth=12, num_heads=12,
    decoder_embed_dim=512, decoder_num_heads=16,
    mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    use_custom_patch=False, 
    in_chans=in_chans, img_size=img_size, audio_exp=True, norm_pix_loss=True) 

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
device = "cuda"

def train_one_epoch(model, dataloader, optimizer, device, epoch=None):
    model.train(True)
    model.to(device)
    optimizer.zero_grad()

    model.epoch = epoch
    
    # Create a tqdm progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        loss, _, _ = model(audio, mask_ratio=0.8)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update the progress bar description with the current loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Close the progress bar
    progress_bar.close()
#%%
train_one_epoch(model=model, dataloader=dataloader, optimizer=optimizer, device="cuda")
#%%

for epoch in range(10):
    train_one_epoch(model=model, dataloader=dataloader, optimizer=optimizer, device="cuda", epoch=epoch)
#%%
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=10)
trainer.fit(model, dataloader)




#%%

import hydra
from omegaconf import DictConfig
from transforms import Transform

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Create the transform
    transform = Transform(cfg)
    
    # Load the dataset
    dataset = load_dataset(
        "ashraq/esc50", 
        cache_dir="./data",
        split="train",
    )

    # Define the transform function
    def transform_function(batch):
        input = []
        for audio in batch["audio"]:
            wav = transform(audio["array"])
            wav = torch.from_numpy(wav)

            target_length = 512
            n_frames = wav.shape[0]
            p = target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                wav = m(wav)
            elif p < 0:
                wav = wav[0:target_length, :]

            input.append(wav.unsqueeze(0))

        return {
            'audio': input,
            'label': torch.Tensor(batch["target"]),
        }

    # Apply the transform to the dataset
    dataset.set_transform(transform_function, output_all_columns=False)

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Your training loop or other processing here
    # ...

if __name__ == "__main__":
    main()
