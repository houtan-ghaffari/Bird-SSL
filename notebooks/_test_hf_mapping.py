
#%%

from datasets import load_dataset, Audio

dataset = load_dataset("marsyas/gtzan", "all", trust_remote_code=True, cache_dir="./data/gtzan")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))

# already waveforms! 
#%%
dataset
#%%
from tqdm import tqdm 

for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"][0]



#dataset["train"][0]["file"]
#%%
dataset["train"][0]["audio"]


#%%
dataset_size_bytes  = dataset["train"].dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")
#%%


from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train", trust_remote_code=True, cache_dir="./data/minds14")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000)) # already waveforms! 

#%%

from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
#%%

sampling_rate = feature_extractor.sampling_rate
sampling_rate = 32_000
#%%

from datasets import Audio

gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

#%%
from datasets import load_from_disk
from datasets import Audio

dataset = load_from_disk("/home/lrauch/projects/birdMAE/data/HSN/HSN_processed_42_cdb073221fc18e3d")

dataset
#%%
dataset = dataset.cast_column("filepath", Audio(sampling_rate=32_000, decode=True, mono=True))

#%%
dataset["train"][0]["filepath"]


#%%
from tqdm import tqdm 

for i in tqdm(range(100)):
    dataset["train"][i]["filepath"]["array"]

#%%

small = dataset["train"].select(range(100))
#%%
small

#%%
from torchaudio.compliance.kaldi import fbank
import torch
fbank_features_test = fbank(
                torch.from_numpy(testarray).unsqueeze(0),
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
)
#%%

def prepare_dataset(batch):
    audio = batch['filepath']
    batch["input_values"] = audio["array"][:160_000]

    batch["label"] = batch["labels"]
    return batch


# Apply the mapping function to the dataset
mapped_dataset = small.map(
    prepare_dataset, 
    remove_columns=small.column_names)
#%%
small[0]["filepath"]["array"]
#%%
from torchaudio.compliance.kaldi import fbank
import torch 
import numpy as np
from torch.nn.functional import pad
from PIL import Image

def prepare_dataset(batch):
    #pad_to_160k = lambda x: pad(x, (0, 160_000 - x.shape[0]), "constant", 0)
    #data = [torch.from_numpy(b["array"][:160_000]) for b in batch["audio"]]
    data = [torch.from_numpy(b["array"]) for b in batch["audio"]]
    #data = [pad_to_160k(d) for d in data]

    imgs = []
    for d in data: 
        img = fbank(
            d.unsqueeze(0),
            htk_compat=True,
            sample_frequency=32_000,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10
        )
        imgs.append(img.T)
    imgs = [Image.fromarray(img.numpy()) for img in imgs]
    
    # del batch['filepath']
    # del batch['detected_events']
    # del batch['start_time']
    # del batch['end_time']
    batch['input_values'] = imgs
    batch["label"] = batch["human_labels"]

    return batch

#%%

from datasets import Audio, load_dataset
import json
from datasets import Sequence, ClassLabel
dataset = load_dataset(
    "agkphysics/AudioSet", 
    cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")

dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
def _one_hot_encode(batch):
    label_list = [y for y in batch["human_labels"]]
    
    # Use numpy instead of torch for caching
    class_one_hot_matrix = np.zeros((len(label_list), 527), dtype=np.float32)
    
    for class_idx, indices in enumerate(label_list):
        class_one_hot_matrix[class_idx, indices] = 1.0
    
    return {"human_labels": class_one_hot_matrix}
with open("/home/lrauch/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
    ontology = json.load(f)
num_classes = len(ontology)
label_names = list(ontology.keys())
class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
dataset = dataset.cast_column("human_labels", class_label)
dataset = dataset.map(_one_hot_encode, batched=True, batch_size=1000, load_from_cache_file=True)

rows_to_remove = [15_759,17_532] #corrupted
all_indices = list(range(len(dataset["train"])))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
dataset["train"] = dataset["train"].select(indices_to_keep)

rows_to_remove = [6_182] #corrupted
all_indices = list(range(len(dataset["test"])))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
dataset["test"] = dataset["test"].select(indices_to_keep)

#%%

dataset["train"][0]["human_labels"]
#%%

dataset

#%%

from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"]
#%%


dataset= dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=True,
    batch_size=100)

#%%

dataset["train"][0]["label"]
#%%

dataset.save_to_disk("./data/audioset_balanced_prepared")

#%%
from datasets import load_dataset, load_from_disk

dataset = load_from_disk("./data/audioset_balanced_prepared")
#%%
import pylab as plt
import numpy as np
plt.imshow(np.array(dataset["train"][0]["input_values"]), cmap='viridis')

#%%
from datasets import Audio, load_dataset

dataset_ = load_dataset(
    "agkphysics/AudioSet", 
    cache_dir="/home/lrauch/projects/birdMAE/data/audioset_balanced")

dataset_ = dataset_.cast_column("audio", Audio(sampling_rate=32_000))
from torchaudio.compliance.kaldi import fbank
import torch
fbank_features_test = fbank(
                torch.from_numpy(dataset_["train"][0]["audio"]["array"]).unsqueeze(0),
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
)
#%%
fbank_features_test.shape

plt.imshow(fbank_features_test.T, cmap='viridis')

#%%

fbank_features_test.T - np.array(dataset["train"][0]["input_values"])
#%%
import matplotlib.pyplot as plt
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    plt.imshow(np.array(dataset["train"][i]["input_values"]), cmap='viridis')
    plt.show()
#%%

np.array(dataset["train"][0]["input_values"])



#%%

from datasets import Audio, load_dataset

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
#%%
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"]

#%%
#%%

dataset

#%%
dataset= dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=True,
    batch_size=500)
#%%
dataset

#%%
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    np.array(dataset["train"][i]["input_values"])
#%%
dataset.save_to_disk("./data/esc50_prepared_n")
#%%

dataset["train"][10]["input_values"]

#%%






dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
    split="train",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
dataset_small = dataset.select(range(100))
dataset_smaller = dataset_small.map(
    prepare_dataset,
    remove_columns=dataset_small.column_names,
    batched=True,
    batch_size=2)
plt.imshow(np.array(dataset_smaller[0]["input_values"]), cmap='viridis')

#%%

# Apply the mapping function to the dataset
mapped_dataset = small.map(
    prepare_dataset,
    remove_columns=small.column_names,
    batched=True,
    batch_size=1)

#%%
import pylab as plt
from torchvision.utils import make_grid
rnd_idx = np.random.randint(0, len(mapped_dataset))
sample = mapped_dataset[rnd_idx]
print(sample)


img = sample['input_values']
# img_max = sample['max']
# img_min = sample['min']
# print(type(img))
# original_img = (img  * (img_max - img_min)) + img_min
# print(original_img.min(), original_img.max())
print(img.min(), img.max())
plt.imshow(img, cmap='viridis')
plt.show()

#%%
from datasets import load_dataset

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
    split="train",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))


#%%

dataset_small = dataset.select(range(100))
#%%

len(dataset_small[0]["audio"]["array"])

#%%

dataset_smaller = dataset_small.map(
    prepare_dataset,
    remove_columns=dataset_small.column_names,
    batched=True,
    batch_size=1)
#%%

dataset_smaller[0]
#%%
np.array(dataset_smaller[0]["input_values"])
#%%
plt.imshow(dataset_smaller[0]["input_values"], cmap='viridis')

#%%



mapped_dataset[:2]



#%%
mapped_dataset[0]["input_values"].shape
#%%
from tqdm import tqdm 

for i in tqdm(range(100)):
    mapped_dataset[i]["filepath"]["array"]
#%%
mapped_dataset._get_cache_file_path("input_values")
#%%
#%%
dataset_size_bytes  = dataset["train"].dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")
#dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))

#%%
from datasets import load_dataset

dataset = load_dataset("DBD-research-group/BirdSet", "XCM", cache_dir="/home/lrauch/projects/birdMAE/data/XCM", num_proc=3)