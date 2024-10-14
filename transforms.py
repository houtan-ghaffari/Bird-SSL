from dataclasses import dataclass
import numpy as np
from transformers.audio_utils import spectrogram, window_function, mel_filter_bank
from omegaconf import DictConfig
import torch 

from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import FrequencyMasking, TimeMasking
from typing import Any, List, Optional, Union

import numpy as np
from transformers import BatchFeature
from transformers import SequenceFeatureExtractor
from transformers.utils import logging, PaddingStrategy
import torch 
import random

logger = logging.get_logger(__name__)

class BaseTransform:
    def __init__(self, 
                 transform_params: DictConfig,         
                 target_length: int,
                 sampling_rate:int,
                 mean: float,
                 std: float,
                 columns: List[str],
                 clip_duration: float
        ):
        self.sampling_rate = sampling_rate  
        self.target_length = target_length 
        self.mean = mean
        self.std = std

        self.columns = columns
        self.clip_duration = clip_duration
        self.fbank_params = transform_params.fbank
        #self.mel_filters = self._init_mel_filters()

        # self.window = window_function(
        #     window_length=self.spectrogram_params.frame_length, 
        #     name=window_params.type, 
        #     periodic=window_params.periodic)  

        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            return_attention_mask=False
        )    

        self.mixup = transform_params.mixup

        if transform_params.freqm:
            self.freqm = FrequencyMasking(freq_mask_param=transform_params.freqm)
        if transform_params.timem:
            self.timem = TimeMasking(time_mask_param=transform_params.timem)
  
    def _init_mel_filters(self):
        mel_filters = mel_filter_bank(**self.mel_filter_params)
        return np.pad(mel_filters, ((0, 1), (0, 0)))

    def _process_waveforms(self, waveforms):
        max_length = int(int(self.sampling_rate) * self.clip_duration)
        waveform_batch = self.feature_extractor(
            waveforms,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        waveform_batch["input_values"] = waveform_batch["input_values"] - waveform_batch["input_values"].mean(axis=1, keepdims=True)
        return waveform_batch 
    
    def _compute_fbank_features(self, waveforms):
        fbank_features = [
            fbank(
                waveform.unsqueeze(0),
                htk_compat=self.fbank_params.htk_compat,
                sample_frequency=self.sampling_rate,
                use_energy=self.fbank_params.use_energy,
                window_type=self.fbank_params.window_type,
                num_mel_bins=self.fbank_params.num_mel_bins,
                dither=self.fbank_params.dither,
                frame_shift=self.fbank_params.frame_shift
            )
            for waveform in waveforms
        ]
        return torch.stack(fbank_features)
    
    def _pad_and_normalize(self, fbank_features):
        difference = self.target_length - fbank_features[0].shape[0]
        if self.target_length > fbank_features.shape[0]:
            m = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank_features = m(fbank_features)

        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        # fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1)
        # fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return fbank_features
    
    def __call__(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)
        fbank_features = self._compute_fbank_features(waveform_batch["input_values"])
        fbank_features = self._pad_and_normalize(fbank_features)
        fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return {
            "audio": fbank_features.unsqueeze(1),
            "label": torch.Tensor(batch[self.columns[1]])
        }

class TrainTransform(BaseTransform):
    
    def cyclic_rolling_start(self, waveforms):
        batch_size, waveform_length = waveforms.shape
        idx = torch.randint(0, waveform_length, (batch_size,), device=waveforms.device)
        arange = torch.arange(waveform_length, device=waveforms.device).unsqueeze(0).expand(batch_size, -1)
        rolled_indices = (arange + idx.unsqueeze(1)) % waveform_length
        rolled_waveforms = waveforms[torch.arange(batch_size).unsqueeze(1), rolled_indices]
        volume_mag = torch.distributions.Beta(10, 10).sample((batch_size, 1)).to(waveforms.device) + 0.5
        waveforms = rolled_waveforms * volume_mag
        
        return waveforms  
    
    def __call__(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)
        waveform_batch["input_values"] = self.cyclic_rolling_start(waveform_batch["input_values"])

        # mixup
        if self.mixup: 
            waveform_batch["input_values"], batch[self.columns[1]] = self._mixup(waveform_batch, batch[self.columns[1]])

        fbank_features = self._compute_fbank_features(waveform_batch["input_values"]) #shape for as: batch, 998, 128
        fbank_features = self._pad_and_normalize(fbank_features) # shape: batch, time(1024), freq(128)
        
        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        if self.freqm: 
            fbank_features = fbank_features.permute(0, 2, 1).unsqueeze(1) # batch, 1, 128, 1024
            fbank_features = torch.stack([self.freqm(feature) for feature in fbank_features])
            fbank_features = torch.stack([self.timem(feature) for feature in fbank_features])
            #fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1) # time, freq
            fbank_features = fbank_features.squeeze(1)  # Remove the channel dimension
            fbank_features = fbank_features.permute(0, 2, 1)  # batch, 1, 1024, 128

        fbank_features = (fbank_features - self.mean) / (self.std * 2) # need: batch, 1024, 128
       
       
        return {
            "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
            "label": torch.Tensor(batch[self.columns[1]]),
        }
    
    def _mixup(self, waveform_batch, labels):
        mixed_audio = []
        mixed_labels = []
        batch_length = len(labels)
        for idx in range(batch_length):
            if random.random() < self.mixup:
                mix_sample_idx = random.randint(0, batch_length - 1)
                mix_lambda = np.random.beta(10, 10)
                mix_waveform = mix_lambda * waveform_batch["input_values"][idx] + (1 - mix_lambda) * waveform_batch["input_values"][mix_sample_idx]

                mix_waveform = mix_waveform - mix_waveform.mean()
                mixed_audio.append(mix_waveform)
                mixed_labels.append([mix_lambda * l1 + (1 - mix_lambda) * l2 for l1, l2 in zip(labels[idx], labels[mix_sample_idx])])
            else:
                mixed_audio.append(waveform_batch["input_values"][idx])
                mixed_labels.append(labels[idx])

        waveform_batch["input_values"] = torch.stack(mixed_audio)
        return torch.stack(mixed_audio), mixed_labels


class EvalTransform(BaseTransform):
    pass

class DefaultFeatureExtractor(SequenceFeatureExtractor):
    """
    A class used to extract features from audio data.

    Attributes
    ----------
    _target_ : str
        Specifies the feature extractor component used in the pipeline.
    feature_size : int
        Determines the size of the extracted features.
    sampling_rate : int
        The sampling rate at which the audio data should be processed.
    padding_value : float
        The value used for padding shorter sequences to a consistent length.
    return_attention_mask : bool
        Indicates whether an attention mask should be returned along with the processed features.
    """
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 32000,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.return_attention_mask = return_attention_mask

    def __call__(
        self,
        waveform: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: int = None,
        truncation: bool = False,
        return_attention_mask: bool = False):
        #return_tensors: str = "pt"):

        waveform_encoded = BatchFeature({"input_values": waveform})

        padded_inputs = self.pad(
            waveform_encoded,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask
        )

        padded_inputs["input_values"] = torch.tensor(
            padded_inputs["input_values"])
        attention_mask = padded_inputs.get("attention_mask")

        if attention_mask is not None:
            padded_inputs["attention_mask"] = attention_mask


        return padded_inputs