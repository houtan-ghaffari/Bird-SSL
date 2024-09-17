#%%
import torch 
from datasets import load_dataset, load_from_disk
import os
import torchaudio
from datasets import Audio
import numpy as np 
from transformers import AutoFeatureExtractor
from torchaudio.compliance.kaldi import fbank
#%%
dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data",
    split="train",
)

#%%
from IPython.display import Audio as IPythonAudio
IPythonAudio(dataset[0]["audio"]["array"],
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
def transform(batch):
    wavs = [cyclic_rolling_start(audio["array"]) for audio in batch["audio"]]
    fbank = fbank(
        wavs, 
        htk_compat=True, 
        sample_frequency=44, 
        use_energy=False,
        window_type='hanning', 
        num_mel_bins=self.melbins,
        dither=0.0, 
        frame_shift=10)
    
    return {
        'audio': wavs,
        'label': torch.Tensor(batch["target"]),
    }
dataset.set_transform(transform, output_all_columns=False)


#%%
dataset[0]["audio"]
IPythonAudio(dataset[0]["audio"],rate=44_100)
#%%
import matplotlib.pyplot as plt
plt.imshow(dataset[0]["audio"])

#%%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
next(iter(dataloader))


#%%



#%%



#%%
dataset[0]["audio"]
#%
cyclic_rolling_start(dataset[0]["audio"])
#%%
dataset[0]
#%%
dataset["train"][10]

#%%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

def train_one_epoch(model, data_loader, device):
