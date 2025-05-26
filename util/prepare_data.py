import pandas as pd
from collections import Counter
import random

from datasets import load_dataset, Audio, DatasetDict
from birdset.datamodule.components.event_mapping import XCEventMapping
from tqdm import tqdm

def smart_sampling( dataset, label_name, class_limit, event_limit):
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="unique_id"
    )
    df = pd.DataFrame(dataset)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="smart sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)

# pretraining data 
def process_pretraining_dataset(
    cache_dir: "str" = ".",
    dataset_name: "str" = "XCL",       
    num_proc: "int" = 1,           
    revision: "str" = "main",
    mapping_batch_size: "int" = 300,
    mapping_num_proc: "int" = 3,  
    save_path: "str" = ".",          
    class_limit: "int" = 500, 
    event_limit: "int" = 2, 
    hf_path: str = "DBD-research-group/BirdSet", 
    audio_sampling_rate: int = 32_000,
    smart_sampling_label_name: str = "ebird_code", 
    final_columns: list = None 
):
    """
    Loads a dataset from Hugging Face, processes its 'train' split (including audio casting,
    event mapping, optional smart sampling, and column selection), and saves it to disk.

    Args:
        cache_dir: Directory for caching Hugging Face datasets.
        dataset_name: The specific configuration or subset of the dataset to load (e.g., "XCL").
        num_proc: Number of processes for `load_dataset`.
        revision: Specific dataset version (e.g., commit hash) to load.
        mapping_batch_size: Batch size for the event mapping step.
        mapping_num_proc: Number of processes for the event mapping step.
        save_path: The full directory path where the processed dataset should be saved.
                   The naming of this path should reflect whether smart sampling was applied
                   (e.g., include class/event limits or 'allevents'), as this function
                   does not modify `save_path` based on sampling parameters.
        class_limit: If provided (not None) along with `event_limit`, smart sampling is applied
                     with this class limit.
        event_limit: If provided (not None) along with `class_limit`, smart sampling is applied
                     with this event limit.
        hf_path: The main path or name of the dataset on Hugging Face Hub.
        audio_sampling_rate: The sampling rate to which audio data will be cast.
        smart_sampling_label_name: The name of the label column used for smart sampling.
        final_columns: A list of column names to retain in the processed dataset. If None,
                       a default list from the original script is used.

    Returns:
        datasets.DatasetDict: The processed dataset, typically containing the 'train' split.

    Raises:
        ValueError: If the loaded dataset does not contain a 'train' split.

    Note:
        This function assumes that `XCEventMapping` class and `smart_sampling` function
        are defined and available in the scope.
    """
    if final_columns is None:
        final_columns = ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]

    print(f"Loading dataset: {hf_path} (configuration: {dataset_name}, revision: {revision})", flush=True)

    ds = load_dataset(
        path=hf_path,
        name=dataset_name,
        cache_dir=cache_dir,
        num_proc=num_proc,
        revision=revision
    )


    if "train" not in ds:
        raise ValueError(f"Dataset {hf_path} (config: {dataset_name}) loaded successfully, but does not contain a 'train' split.")
    train_data = ds["train"]

    print(f"Casting 'audio' column for 'train' split to {audio_sampling_rate} Hz, mono, decode=False.", flush=True)
    train_data = train_data.cast_column(
        column="audio",
        feature=Audio(
            sampling_rate=audio_sampling_rate,
            mono=True,
            decode=False, 
        ),
    )

    mapper = XCEventMapping()

    print(f"Performing event mapping on 'train' split (batch size: {mapping_batch_size}, num_proc: {mapping_num_proc}).", flush=True)
    train_data = train_data.map(
        mapper, 
        remove_columns=["audio"], 
        batched=True,
        batch_size=mapping_batch_size,
        num_proc=mapping_num_proc,
        desc=f"Event mapping for {dataset_name} train split"
    )

    if class_limit is not None or event_limit is not None:
        print(f"Applying smart sampling to 'train' split: class_limit={class_limit}, event_limit={event_limit}, label='{smart_sampling_label_name}'", flush=True)
        train_data = smart_sampling(
           dataset=train_data,
           label_name=smart_sampling_label_name,
           class_limit=class_limit,
           event_limit=event_limit,
        )
    else:
        print("Skipping smart sampling for 'train' split.", flush=True)

    print(f"Selecting final columns for 'train' split: {final_columns}", flush=True)
    train_data = train_data.select_columns(columns=final_columns)


    processed_ds_dict = DatasetDict({
        "train": train_data
    })

    print(f"Saving processed dataset to: {save_path}", flush=True)
    processed_ds_dict.save_to_disk(save_path)
    print(f"Dataset successfully saved to {save_path}", flush=True)



# downstream data  complete
import pandas as pd
from collections import Counter
import random
from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

names = ["PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE"]

for name in names:
    print(f"Loading {name}", flush=True)
    dataset = load_dataset("DBD-research-group/BirdSet", name, num_proc=5, cache_dir=f"/scratch/birdset/{name}")
    print(f"Loaded {name}", flush=True)


for name in names:
    print(f"preparing {name}", flush=True)
    dm = BirdSetDataModule(
        dataset= DatasetConfig(
            data_dir=f"/scratch/birdset/{name}", # specify your data directory!
            hf_path='DBD-research-group/BirdSet',
            hf_name=name,
            n_workers=1,
            val_split=0.0001,
            task="multilabel",
            classlimit=500, #limit of samples per class
            eventlimit=5, #limit of events that are extracted for each sample
            sampling_rate=32_000,
        ),
    )
    dm.prepare_data()
    print(dm.disk_save_path)

# fewshot data 
from datasets import DatasetDict, Dataset
import random
from birdset.datamodule.components.event_mapping import XCEventMapping
import soundfile as sf
import torch

class BaseCondition:

    def __call__(self, dataset: Dataset, idx:int , **kwds) -> bool:
        return True


class LenientCondition(BaseCondition):

    def __call__(self, dataset: Dataset, idx: int, **kwds):
        """
        This condition allows files up to 10s but only if one bird occurence is in the file.
        """
        file_info = sf.info(dataset[idx]["filepath"])
        if file_info.duration <= 20 and (not dataset[idx]["ebird_code_secondary"]):
            return True

class StrictCondition(BaseCondition):

    def __call__(self, dataset: Dataset, idx: int, **kwds):
        """
        This condition only allows files that up to 5s long so that no event detection has to occur when sampling.
        """
        file_info = sf.info(dataset[idx]["filepath"])
        if file_info.duration <= 5:
            return True

def one_hot_encode_batch(batch, num_classes):
    """
    Converts integer class labels in a batch to one-hot encoded tensors.
    """
    label_list = batch["labels"]
    batch_size = len(label_list)
    one_hot = torch.zeros((batch_size, num_classes), dtype=torch.float32)
    for i, label in enumerate(label_list):
        one_hot[i, label] = 1
    return {"labels": one_hot}

def create_few_shot_subset(dataset: DatasetDict, few_shot: int=5, data_selection_condition: BaseCondition=StrictCondition(), fill_up: bool=False, random_seed: int=None) -> DatasetDict:
    """
    This method creates a subset of the given datasets train split with at max `few_shot` samples per label in the dataset split.
    The samples are chosen based on the given condition. If there are more than `few_shot` samples for a label `few_shot`
    random samples are chosen. If exactly `few_shot` samples per label are wanted, `fill_up` should be set to `True`.
    After the samples that pass the condition are added to the subset, this will randomly fill up the unfullfilled labels
    with their respective samples from the given dataset split without regard for the condition.

    Args:
        dataset (DatasetDict): A Huggingface "datasets.DatasetDict" object. A few-shot subset will be created for the `train` split.
        few_shot (int): The number of samples each label can have. Default is 5.
        data_selection_condition (ConditionTemplate): A condition that defines which recordings should be included in the few-shot subset.
        fill_up (bool): If True, labels for which not enough samples can be extracted with the given condition will be supplemented with
          random samples from the dataset. Default is False.
        random_seed (int): The seed with which the random sampler is seeded. If None, no seeding is applied. Default is None.
    Returns:
        DatasetDict: A Huggingface `datasets.DatasetDict` object where the test split is return as it was given and the train
        split is replaced with the few-shot subset of the given train split.
    """
    if random_seed != None:
        print(f"Set random seed to {random_seed}.")
        random.seed(random_seed)
    train_split = dataset["train"]
    num_classes = train_split.features["ebird_code"].num_classes

    print("Applying condition to training data.")
    satisfying_recording_indeces = []
    for i in range(len(train_split)):
        if data_selection_condition(train_split, i):
            satisfying_recording_indeces.append(i)

    print("Mapping satisfying recordings.")
    all_labels = set(train_split["ebird_code"])
    primary_samples_per_label, leftover_samples_per_label = _map_recordings_to_samples(
        train_split,
        all_labels,
        satisfying_recording_indeces
    )

    print("Selecting samples for subset.")
    selected_samples = []
    unfullfilled_labels = {}
    for label, samples in primary_samples_per_label.items():
        num_primary_samples = len(samples)
        num_leftover_samples = len(leftover_samples_per_label[label])
        if (num_primary_samples + num_leftover_samples) < few_shot:
            selected_samples.extend(samples)
            selected_samples.extend(leftover_samples_per_label[label])
            unfullfilled_labels[label] = few_shot - (num_primary_samples + num_leftover_samples)
        elif num_primary_samples < few_shot:
            selected_samples.extend(samples)
            selected_samples.extend(random.sample(leftover_samples_per_label[label], k=(few_shot - num_primary_samples)))
        else:
            selected_samples.extend(random.sample(samples, few_shot))

    if fill_up:
        print("Filling up labels.")
        unused_recordings = set(range(len(train_split))).difference(satisfying_recording_indeces)
        unused_primary, unused_leftover = _map_recordings_to_samples(
            train_split,
            all_labels,
            unused_recordings
        )

        fill_up_samples = []
        for label, count in unfullfilled_labels.items():
            num_primary_samples = len(unused_primary[label])
            num_leftover_samples = len(unused_leftover[label])
            if num_primary_samples < count:
                fill_up_samples.extend(unused_primary[label])
                # if there are not enough samples in the dataset the min() has to be taken to avoid errors.
                fill_up_samples.extend(random.sample(unused_leftover[label], k=min((count - num_primary_samples), num_leftover_samples)))
            else:
                fill_up_samples.extend(random.sample(unused_primary[label], count))
        selected_samples.extend(fill_up_samples)

    dataset = DatasetDict({"train": Dataset.from_list(selected_samples), "test": dataset["test_5s"]})


    print("Selecting relevant columns and renaming...", flush=True)
    columns_to_keep = ["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"]

    dataset = DatasetDict({
    split: dataset[split].select_columns(columns_to_keep).rename_column("ebird_code_multilabel", "labels")
    for split in dataset.keys()
    })

    print("Applying one-hot encoding to labels...", flush=True)
    dataset = dataset.map(lambda batch: one_hot_encode_batch(batch, num_classes), batched=True)

    return dataset


def _map_recordings_to_samples(train_split: Dataset, all_labels: set, recording_indeces: list):
    """
    This method uses the XCEventMapping to extract samples from the recordings. It also splits
    the extracted samples into primary and leftover. Every recording has exaclty one primary sample,
    which is chosen randomly. All samples that are not a primary sample for a recording are saved as
    leftover samples.
    """
    mapper = XCEventMapping()
    primary_samples_per_label = {label: [] for label in all_labels}
    leftover_samples_per_label = {label: [] for label in all_labels}
    for idx in recording_indeces:
        mapped_batch = mapper({key: [value] for key, value in train_split[idx].items()})
        mapped_batch = _remove_duplicates(mapped_batch)
        # in cases where a recording produces multiple samples, choose one as the main sample
        # to prioritise the selection of samples from differing recordings.
        num_samples = len(mapped_batch["filepath"])
        primary_sample = random.choice(range(num_samples))
        for i in range(num_samples):
            sample = {key: mapped_batch[key][i] for key in mapped_batch.keys()}
            if i == primary_sample:
                primary_samples_per_label[sample["ebird_code"]].append(sample)
            else:
                leftover_samples_per_label[sample["ebird_code"]].append(sample)
    return primary_samples_per_label, leftover_samples_per_label


def _remove_duplicates(batch: dict[str, ]):
    """
    This method removes basic duplicates samples from a batch. These are samples that
    are entirely included in another sample in the same batch. It only works correctly if all
    samples in the batch are from the same recording.
    """
    removable_idx = set()
    num_samples = len(batch["filepath"])
    for b_idx in range(num_samples):
        for other_sample in range(b_idx + 1, num_samples):
            event_one = batch["detected_events"][b_idx]
            event_two = batch["detected_events"][other_sample]
            if event_one[0] < event_two[0] and event_one[1] > event_two[1]:
                removable_idx.add(other_sample)
            elif event_two[0] < event_one[0] and event_two[1] > event_one[1]:
                removable_idx.add(b_idx)

    new_batch = {}
    for key in batch.keys():
        new_batch[key] = []
        for b_idx in range(num_samples):
            if b_idx not in removable_idx:
                new_batch[key].append(batch[key][b_idx])

    return new_batch



import os
from datasets import load_dataset
import sys
import os

# Define dataset names and optional revisions.
datasets_info = {
     "HSN": {},
     "POW": {},
     "NES": {},
     "PER": {},
     "SNE": {},
     "SSW": {},
     "UHH": {},
     "NBP": {},
}

# Define shot levels and seeds.
shot_numbers = [1, 5, 10] 
seeds = [1, 2, 3]

# Base directory where the few-shot subsets will be saved.
base_save_path = "/scratch/birdset"

for ds_name, params in datasets_info.items():
    revision = params.get("revision")
    print(f"Loading dataset {ds_name}...")
    # Load dataset with revision if provided.
    if revision:
        ds = load_dataset("DBD-research-group/BirdSet", ds_name, num_proc=1, revision=revision,
                           cache_dir=os.path.join(base_save_path, ds_name))
    else:
        ds = load_dataset("DBD-research-group/BirdSet", ds_name, num_proc=1,
                          cache_dir=os.path.join(base_save_path, ds_name))

    # Compute NUM_CLASSES from the dataset's ClassLabel feature.
    NUM_CLASSES = ds["train"].features["ebird_code"].num_classes
    print(f"{ds_name} has {NUM_CLASSES} classes.")

    for shot in shot_numbers:
        for seed in seeds:
            print(f"Creating {shot}-shot subset for {ds_name} with seed {seed}...")
            few_shot_ds = create_few_shot_subset(
                ds,
                few_shot=shot,
                data_selection_condition=LenientCondition(),
                fill_up=False,
                random_seed=seed
            )
            # Define the saving path.
            save_dir = os.path.join(base_save_path, ds_name, f"{ds_name}_{shot}shot_{seed}")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            few_shot_ds.save_to_disk(save_dir)
            print(f"Saved {ds_name} {shot}-shot, seed {seed} subset to {save_dir}")