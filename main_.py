#%%
from datasets import load_dataset
#%%
#%%
import torch 
from datasets import load_dataset, load_from_disk, save_to_disk
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


#%%

from datasets import load_dataset 

dataset = load_dataset("agkphysics/AudioSet", cache_dir="/home/lrauch/projects/birdMAE/data/agkphysics/AudioSet")

#%%

dataset["test"][0]
#%%

dataset

#%%

from datasets import load_dataset , Audio


dataset = load_dataset("agkphysics/AudioSet", cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")


#%%

train_data = dataset["train"]
train_data.set_format("numpy", columns=["audio","human_labels"], output_all_columns=False)
train_data = train_data.cast_column("audio", Audio(sampling_rate=32_000, mono=True, decode=True))

#%%
from tqdm import tqdm

#for row in tqdm(train_data.select(range(10))): 
for row in tqdm(train_data): 
    try:
        row["audio"]["array"][0]
    except:
        print(f"error: {row}")
        break
#%%

train_data[15759]["audio"]["array"]

#%%

train_data[15759]["human_labels"]

# %%

row_to_remove = 15_759  # Example: remove the 6th row (index 5)

# Step 3: Create a list of all indices except the one to remove
all_indices = list(range(len(train_data)))
indices_to_keep = [i for i in all_indices if i != row_to_remove]


rows_to_remove= 17_532

#%%

indices_to_keep = list(range(len(train_data))[17_532+1:])
# Step 4: Use select to keep only the rows you want
new_dataset = train_data.select(indices_to_keep)

#%%

new_dataset

#%%

for row in tqdm(new_dataset): 
    try:
        row["audio"]["array"][0]
    except:
        print(f"error: {row}")
        break
#%%

train_data[17532]["audio"]["array"]

#%%

from datasets import load_dataset , Audio

dataset = load_dataset("agkphysics/AudioSet", cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")

test_data = dataset["test"]
test_data.set_format("numpy", columns=["audio","human_labels"], output_all_columns=False)
test_data = test_data.cast_column("audio", Audio(sampling_rate=32_000, mono=True, decode=True))


#%%

test_data[6182]["audio"]["array"]
#%%
from tqdm import tqdm

#for row in tqdm(train_data.select(range(10))): 
for row in tqdm(test_data): 
    try:
        row["audio"]["array"][0]
    except:
        print(f"error: {row}")
        break

#%%
test_ = 6182

indices_to_keep = list(range(len(test_data))[6182+1:])
# Step 4: Use select to keep only the rows you want
new_dataset = test_data.select(indices_to_keep)

for row in tqdm(new_dataset): 
    try:
        row["audio"]["array"][0]
    except:
        print(f"error: {row}")
        break
#%%
from datasets import load_dataset , Audio

dataset = load_dataset("agkphysics/AudioSet", cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")

test_data = dataset["test"]
test_data.set_format("numpy", columns=["audio","human_labels"], output_all_columns=False)
test_data = test_data.cast_column("audio", Audio(sampling_rate=32_000, mono=True, decode=True))

#%%
test_data
#%%
unique_classes = set()

# Loop through the dataset
for sample["human_labels"] in test_data:
    # Assuming the labels are in a field called 'labels', which is a list of lists with strings
    for label in sample['human_labels']:
        unique_classes.update(label)  # Add all labels in the current sample to the set

# Print the number of unique classes
print(f"Number of unique classes: {len(unique_classes)}")

# Optional: Print the unique classes
print("Unique classes:", unique_classes)


#%% 

# preprocessing 
import json 
import os 
import json
from datetime import datetime
from omegaconf import DictConfig
from datasets import load_dataset, Audio, ClassLabel, Sequence
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl 
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np 

from configs import Config, DataModuleConfig, ModuleConfig
from transforms import TrainTransform, EvalTransform
#%%
dataset = load_dataset(
    "agkphysics/AudioSet", split="train",cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced"
)

#%%

dataset


#%%
columns = ["audio", "human_labels"]
def _one_hot_encode(batch):
    label_list = [y for y in batch[columns[1]]]
    
    # Use numpy instead of torch for caching
    class_one_hot_matrix = np.zeros((len(label_list), num_classes), dtype=np.float32)
    
    for class_idx, indices in enumerate(label_list):
        class_one_hot_matrix[class_idx, indices] = 1.0
    
    return {columns[1]: class_one_hot_matrix}



with open("/home/lrauch/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
    ontology = json.load(f)
num_classes = len(ontology)
label_names = list(ontology.keys())
class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
dataset = dataset.cast_column("human_labels", class_label)
dataset = dataset.map(_one_hot_encode, batched=True, batch_size=1000, load_from_cache_file=True)

rows_to_remove = [15_759,17_532] #corrupted
all_indices = list(range(len(dataset)))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
dataset = dataset.select(indices_to_keep)

#%%

dataset["human_labels"][0]

from datasets import DatasetDict

dataset_dict = DatasetDict({"train": dataset})

#%%

dataset_dict

#%%
test_data = load_dataset(
    "agkphysics/AudioSet", split="test",cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced"
)
#%%

test_data

#%%
with open("/home/lrauch/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
    ontology = json.load(f)
num_classes = len(ontology)
label_names = list(ontology.keys())
class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
test_data = test_data.cast_column("human_labels", class_label)
test_data = test_data.map(_one_hot_encode, batched=True, batch_size=1000)

rows_to_remove = [6_182] #corrupted
all_indices = list(range(len(test_data)))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
test_data = test_data.select(indices_to_keep)
#%%
test_data
#%%
dataset_dict = DatasetDict({"train": dataset, "test": test_data})


#%%


dataset_dict

#%%

dataset_dict.save_to_disk("/home/lrauch/projects/birdMAE/data/audioset_balanced/saved_to_disk")



#%%
from datasets import load_from_disk
load_from_disk("/home/lrauch/projects/birdMAE/data/audioset_balanced/saved_to_disk", split="train")

#%%
train_dataset = load_dataset(
    "agkphysics/AudioSet", split="train",cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced"
)

train_dataset.set_format("numpy", columns=["audio","human_labels"], output_all_columns=False)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=32_000, mono=True, decode=True))

#%%

import IPython.display as ipd

# Get the audio array
audio_array = train_dataset[67]["audio"]["array"]

# Get the sampling rate (assuming it's 32000 as mentioned earlier)
sampling_rate = 32000

# Play the audio
ipd.display(ipd.Audio(audio_array, rate=sampling_rate))


#%%
train_dataset[100]["human_labels"]


#%%
from torchaudio.compliance.kaldi import fbank

fbank_features = fbank(
                torch.from_numpy(train_dataset[67]["audio"]["array"]).unsqueeze(0),
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
)

fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1)
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the spectrogram
im = ax.imshow(fbank_features.T, aspect='auto', origin='lower', interpolation='nearest')

# Set the title and labels
ax.set_title('Mel Spectrogram')
ax.set_xlabel('Time')
ax.set_ylabel('Mel Frequency Bin')

# Add a colorbar
plt.colorbar(im, ax=ax, format='%+2.0f dB')

# Show the plot
plt.tight_layout()

plt.show()
#%%

fbank_features.shape

#%%

from datasets import load_dataset, Audio 


dataset = load_dataset("DBD-research-group/BirdSet", "HSN", cache_dir="/home/lrauch/projects/birdMAE/data/HSN", num_proc=5)

#%%
dataset["train"]["length"][0]
#%%

dataset["train"][0]["audio"]


#%%
#from birdset.datamodule.base_datamodule import DatasetConfig

from birdset.configs.datamodule_configs import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

#%%
DatasetConfig()

#%%

from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_from_disk

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir="/home/lrauch/projects/birdMAE/data/HSN", # specify your data directory!
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500, #limit of samples per class 
        eventlimit=5, #limit of events that are extracted for each sample
        sampling_rate=32000,
    ),
)

#%%

dm.prepare_data()

#%%

ds = load_from_disk(dm.disk_save_path)

#%%

ds["train"][0]

#%%
ds["train"][1]
#%%

ds

#%%

dm.disk_save_path
#%%

# Alternatively, we can create an instance and print its __dict__
birdset_dm = BirdSetDataModule()
print(birdset_dm.__dict__.keys())


#%%

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper

#%%

BirdSetTransformsWrapper(

)

#%%

from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.configs.datamodule_configs import DatasetConfig, LoaderConfig, LoadersConfig

#%%
LoadersConfig()
#%%

dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir="/home/lrauch/projects/birdMAE/data/HSN", # specify your data directory!
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500, #limit of samples per class 
        eventlimit=5, #limit of events that are extracted for each sample
        sampling_rate=32000,
    ),
    loaders=LoadersConfig(), # only utilized in setup
    transforms=BirdSetTransformsWrapper() # set_transform after .setup
)

#%%
dm.prepare_data()
#%%

dm.prepare_data()

#%%

dataset_config = DatasetConfig(
    data_dir='data_birdset/HSN', # specify your data directory!
    hf_path='DBD-research-group/BirdSet',
    hf_name='HSN',
    n_classes=21,
    n_workers=3,
    val_split=0.2,
    task="multilabel", # one-hot-encode classes
    classlimit=500, #limit of samples per class 
    eventlimit=5, #limit of events that are extracted for each sample
    sampling_rate=32000,
)

#%%

loaders_config = LoadersConfig(
    train = LoaderConfig(
        batch_size=32,
        shuffle=True,
        num_workers=1, 
        pin_memory=True,
    ),
    valid = LoaderConfig(
        batch_size=64
        shuffle=False,
    ),
    test = LoaderConfig(
        batch_size=64,
        shuffle=False
    )
)

#%%
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor
from birdset.datamodule.components.transforms import PreprocessingConfig
from birdset.datamodule.components.augmentations import NoCallMixer, PowerToDB
from birdset.datamodule.components.resize import Resizer
import torch_audiomentations
from torchaudio.transforms import Spectrogram, MelScale
import torchvision


wrapper = BirdSetTransformsWrapper(
    task="multilabel",
    sampling_rate=32_000,
    model_type="vision",
    spectrogram_augmentations=[],
    waveform_augmentations=[],
    decoding=EventDecoding(
        min_len=0,
        max_len=5,
        sampling_rate=32_000
    ),
    feature_extractor=DefaultFeatureExtractor(
        feature_size=1,
        padding_value=0.0,
        return_attention_mask=False,
    ),
    max_length=5,
    nocall_sampler=None,
    preprocessing=PreprocessingConfig(
        spectrogram_conversion=Spectrogram(
            n_fft=1024,
            hop_length=320,
            power=2.0),
        melscale_conversion=MelScale(
            n_mels=128,
            sample_rate=32_000,
            n_stft=513
        ),
        dbscale_conversion=PowerToDB(),
        resizer=Resizer(
            db_scale=True, 
            target_height=128, 
            target_width=1024),
        
        normalize_spectrogram=True,
        normalize_waveform=None,
        mean=-4.268,
        std=-4.569
    )
)

#%%

loaders_config

#%%





# Instantiate the BirdSetDataModule
birdset_dm = BirdSetDataModule(
    dataset= dataset_config,
    loaders = loaders_config,
    transforms = wrapper
)


birdset_dm.dataset_config

#%%

birdset_dm.loaders_config


birdset_dm.transforms_config



#%%

from birdset.configs.datamodule_configs import DatasetConfig, LoadersConfig
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from datasets import load_from_disk

# initiate the data module
dm = BirdSetDataModule(
    dataset= DatasetConfig(
        data_dir="/home/lrauch/projects/birdMAE/data/HSN", # specify your data directory!
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        n_workers=3,
        val_split=0.05,
        task="multilabel",
        classlimit=500, #limit of samples per class 
        eventlimit=5, #limit of events that are extracted for each sample
        sampling_rate=32000,
    ),
    loaders=LoadersConfig(), # only utilized in setup
    transforms=BirdSetTransformsWrapper() # set_transform after .setup
)

# prepare the data
dm.prepare_data()
#%%
dm.disk_save_path
#%%
ds = load_from_disk(dm.disk_save_path)
#%%

ds["train"][0]
#%%
ds["test"][0]
#%%
from birdset.datamodule.components.event_decoding import EventDecoding

event_decoder = EventDecoding(
    min_len=1,
    max_len=5,
    sampling_rate=32_000,
    extension_time=8,
    extracted_interval=5
)

#%%

event_decoder(ds["train"][0])
#%%

ds["train"][1]


int(102.656*32_000)
#%%
min_len = 1
max_len = 5

def load_audio(sample):
    path = sample["filepath"]
    start = sample["detected_events"][0]
    end = sample["detected_events"][1]

    file_info = sf.info(path)
    sr = file_info.samplerate

    if start is not None and end is not None:
        if end - start < min_len:  
            end = start + min_len
        if max_len and end - start > max_len:
            end = start + max_len
        start, end = int(start * sr), int(end * sr)
    if not end:
        end = int(max_len * sr)

    audio, sr = sf.read(path, start=start, stop=end)

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return audio, sr
#%%

audio, _ = load_audio(ds["train"][11])




from IPython.display import Audio as IPythonAudio
IPythonAudio(audio,
             rate=32_000)

#%%

def load_audio(sample, min_len, max_len, sampling_rate):
    path = sample["filepath"]
    start = sample["detected_events"][0]
    end = sample["detected_events"][1]

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration

    if start is not None and end is not None:
        event_duration = end - start
        
        if event_duration < min_len:
            # Calculate how much we need to extend on each side
            extension = (min_len - event_duration) / 2
            
            # Try to extend equally on both sides
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            # If we couldn't extend fully on one side, try to extend more on the other side
            if new_start == 0:
                new_end = min(total_duration, new_end + (start - new_start))
            elif new_end == total_duration:
                new_start = max(0, new_start - (new_end - end))
            
            start, end = new_start, new_end

        if end - start > max_len:
            # If longer than max_len, take the first 5 seconds of the event
            end = min(start + 5, total_duration)
            if end - start > max_len:
                end = start + max_len

        start, end = int(start * sr), int(end * sr)
    else:
        # If start and end are not provided, load the first max_len seconds
        start, end = 0, int(min(max_len, total_duration) * sr)

    audio, sr = sf.read(path, start=start, stop=end)

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return audio, sr
#%%
audio, _ = load_audio(ds["test"][11], min_len=5, max_len=5, sampling_rate=32_000)
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio, rate=32_000)

#%%

ds["test"][11]
#%%
import soundfile as sf
import librosa 

sampling_rate = 32_000
audio, sr = sf.read(
    file=ds["train"][11]["filepath"],
    start=int(ds["train"][11]["detected_events"][0] * sampling_rate),
    stop=int(ds["train"][11]["detected_events"][1] * sampling_rate)
)
if audio.ndim != 1:
    audio = audio.swapaxes(1, 0)
    audio = librosa.to_mono(audio)

#%%
audio.shape

#%%

audio_.shape

#%%

audio
#%%


def load_audio(sample, min_len, max_len, sampling_rate):
    path = sample["filepath"]
    
    if sample["detected_events"] is not None:
        start = sample["detected_events"][0]
        end = sample["detected_events"][1]
        event_duration = end - start
        
        if event_duration < min_len:
            extension = (min_len - event_duration) / 2
            
            # try to extend equally 
            new_start = max(0, start - extension)
            new_end = min(total_duration, end + extension)
            
            if new_start == 0:
                new_end = min(total_duration, new_end + (start - new_start))
            elif new_end == total_duration:
                new_start = max(0, new_start - (new_end - end))
            
            start, end = new_start, new_end

        if end - start > max_len:
            # if longer than max_len
            end = min(start + max_len, total_duration)
            if end - start > max_len:
                end = start + max_len
    else:
        start = sample["start_time"]
        end = sample["end_time"]

    file_info = sf.info(path)
    sr = file_info.samplerate
    total_duration = file_info.duration

    start, end = int(start * sr), int(end * sr)
    audio, sr = sf.read(path, start=start, stop=end)

    if audio.ndim != 1:
        audio = audio.swapaxes(1, 0)
        audio = librosa.to_mono(audio)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return audio, sr


#%%

audio, _ = load_audio(ds["test"][30], min_len=5, max_len=5, sampling_rate=32_000)
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio, rate=32_000)

#%%

   import torch 
    datamodule.setup("fit")
    loader = torch.utils.data.DataLoader(dataset=datamodule.train_data,batch_size=50)

    import torch
    from tqdm import tqdm

    # Initialize variables for calculating mean
    total_sum = 0.0
    total_count = 0

    # First pass to calculate the global mean
    for batch in tqdm(loader):
        inputs = batch["audio"]  # Assuming the inputs are in this key
        total_sum += inputs.sum().item()
        total_count += torch.numel(inputs)  # Get the total number of elements in the batch

# Calculate the global mean
    global_mean = total_sum / total_count
    print(f"Global Mean: {global_mean}")

    sum_of_squares = 0.0

# Second pass to calculate the variance
    for batch in tqdm(loader):
        inputs = batch["audio"]
        
        # Subtract the global mean and square the result, then sum
        sum_of_squares += ((inputs - global_mean) ** 2).sum().item()

    # Calculate the global variance and standard deviation
    global_variance = sum_of_squares / total_count
    global_std = torch.sqrt(torch.tensor(global_variance)).item()

    print(f"Global Std Dev: {global_std}")


#%%%

from datasets import load_dataset, Audio, save_to_disk
#%%
import datasets 
#%%
!pip install datasets
#%%
datasets.save_to_disk()
#%%

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
    split="train",
)
#840MB
#%%
dataset
#%%
from transforms import DefaultFeatureExtractor
feature_extractor = DefaultFeatureExtractor(
    feature_size=1,
    sampling_rate=32_000,
    padding_value=0.0,
    return_attention_mask=False
)    


#%%
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays)
    return inputs

mapped_dataset = dataset.map(
    preprocess_function, 
    remove_columns=dataset.column_names,
    batched=True)

#%%
mapped_dataset.save_to_disk("./data/esc50_mapped")


#%%
from datasets import load_from_disk
dataset = load_from_disk("./data/esc50_mapped")
#%%

dataset["input_values"][0].shape
#%%

mapped_dataset["input_values"][0].shape

#%%

from datasets import save_to_disk
#%%
mapped_dataset._get_cache_file_path("input_values")

#%%

dataset["audio"][0]
#%%

dataset_size_bytes  = dataset.dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")


#%%%

dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
#%%

def prepare_dataset(batch):
    audio = batch['audio']
    batch["input_values"] = audio["array"]

    batch["labels"] = batch["target"]
    return batch


# Apply the mapping function to the dataset
mapped_dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names,
    num_proc=2)
#%%

len(mapped_dataset["input_values"][1])

#%%%
mapped_dataset["audio"][0]["array"].shape
#%%
#%%
a = mapped_dataset["audio_array"][0]
#%%

len(mapped_dataset["audio_array"][0])


#%%

dataset["audio"][0]
#%%


dataset._get_cache_file_path("audio")
#842 MB

#%%

dataset._get_cache_file_path("audio")


#%%%

dataset = load_dataset(
    "agkphysics/AudioSet",
      cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")
#%%

dataset_size_bytes  = dataset["train"].dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")