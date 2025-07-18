#!/usr/bin/env python
import os
from datasets import DatasetDict, Dataset, load_dataset
import random
from birdset.datamodule.components.event_mapping import XCEventMapping
import soundfile as sf
import torch
import argparse 

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

CONDITION_MAP = {
    "strict": StrictCondition(),
    "lenient": LenientCondition(),
}


def process_fewshot(
    dataset_names: list[str],
    shot_numbers: list[int],
    seeds: list[int],
    condition: str,
    cache_dir_base: str,
):
    cond = CONDITION_MAP[condition.lower()]
    for ds_name in dataset_names:
        cache_dir = os.path.join(cache_dir_base, ds_name)
        print(f"Loading {ds_name} …", flush=True)
        ds = load_dataset("DBD-research-group/BirdSet", ds_name, cache_dir=cache_dir, num_proc=1)

        for shot in shot_numbers:
            for seed in seeds:
                tag = f"{ds_name}_{shot}shot_{seed}"
                print(f"Creating {tag} with {condition} condition …", flush=True)
                subset = create_few_shot_subset(
                    ds,
                    few_shot=shot,
                    data_selection_condition=cond,
                    fill_up=False,
                    random_seed=seed,
                )
                save_dir = os.path.join(cache_dir_base, ds_name, tag)
                subset.save_to_disk(save_dir)
                print(f"✔ Saved to {save_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Few-shot BirdSet subset creator")
    parser.add_argument("--dataset-names", nargs="+", default=["PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE"])
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10], help="k values for k-shot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--condition", choices=["strict", "lenient"], default="lenient")
    parser.add_argument("--cache-dir-base", type=str, default="/scratch/birdset")
    args = parser.parse_args()

    process_fewshot(args.dataset_names, args.shots, args.seeds, args.condition, args.cache_dir_base)
