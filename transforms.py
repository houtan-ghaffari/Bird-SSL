from dataclasses import dataclass
import numpy as np
from transformers.audio_utils import spectrogram, window_function, mel_filter_bank
from omegaconf import DictConfig
import torch 

from torchaudio.compliance.kaldi import fbank
from typing import Any, List, Optional, Union

import numpy as np
from transformers import BatchFeature
from transformers import SequenceFeatureExtractor
from transformers.utils import logging, PaddingStrategy
import torch 

logger = logging.get_logger(__name__)

class Transform:
    def __init__(self, 
                 mel_params: DictConfig,
                 spectrogram_params: DictConfig,
                 window_params: DictConfig,             
                 target_length: int,
                 mean: float,
                 std: float
        ):
        self.sampling_rate = mel_params.sampling_rate  
        self.target_length = target_length 
        self.mean = mean
        self.std = std
        self.spectrogram_params = spectrogram_params  
        self.mel_filter_params = mel_params    

        self.mel_filters = self._init_mel_filters()
        self.window = window_function(
            window_length=self.spectrogram_params.frame_length, 
            name=window_params.type, 
            periodic=window_params.periodic)

        self.target_length = target_length    

        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            return_attention_mask=False
        )    

    # def cyclic_rolling_start(self, waveform):
    #     idx = np.random.randint(0, len(waveform))
    #     rolled_waveform = np.roll(waveform, idx)
    #     volume_mag = np.random.beta(10,10) + 0.5
    #     waveform = rolled_waveform * volume_mag
    #     waveform = waveform - waveform.mean()
    #     return waveform

    def cyclic_rolling_start(self, waveforms):
        # waveforms shape: (batch_size, waveform_length)
        batch_size, waveform_length = waveforms.shape
    
        # Generate random indices for each waveform in the batch
        idx = torch.randint(0, waveform_length, (batch_size,), device=waveforms.device)
    
        # Create a range tensor for each waveform
        arange = torch.arange(waveform_length, device=waveforms.device).unsqueeze(0).expand(batch_size, -1)
    
        # Calculate the rolled indices
        rolled_indices = (arange + idx.unsqueeze(1)) % waveform_length
    
        # Use advanced indexing to roll each waveform
        rolled_waveforms = waveforms[torch.arange(batch_size).unsqueeze(1), rolled_indices]
    
        # Generate random volume magnitudes for the batch
        volume_mag = torch.distributions.Beta(10, 10).sample((batch_size, 1)).to(waveforms.device) + 0.5
    
        # Apply volume adjustment
        waveforms = rolled_waveforms * volume_mag
        
        return waveforms    

    def _init_mel_filters(self):
        mel_filters = mel_filter_bank(**self.mel_filter_params)
        return np.pad(mel_filters, ((0, 1), (0, 0)))
    	
    # def __call__(self, batch):
    #     waveforms = torch.stack([torch.from_numpy(audio["array"]) for audio in batch["audio"]])
    #     waveforms = waveforms - waveforms.mean(axis=1, keepdims=True)
    #     waveforms = self.cyclic_rolling_start(waveforms)
    #     fbank = [fbank(w, htk_compat=True, sample_frequency=self.sampling_rate, use_energy=False,
    #             window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    #         for w in waveforms]
        
    #     return fbank

    def __call__(self, batch):

        waveform_batch = [audio["array"] for audio in batch["audio"]]
        max_length = int(int(self.sampling_rate) * 5)
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        waveform_batch["input_values"] = waveform_batch["input_values"] - waveform_batch["input_values"].mean(axis=1, keepdims=True)
        waveform_batch["input_values"] = self.cyclic_rolling_start(waveform_batch["input_values"]) 

        fbank_features= [
            fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sampling_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            )
            for waveform in waveform_batch["input_values"]
        ]

        fbank_features = torch.stack(fbank_features) # 498 x 128

        difference = self.target_length - fbank_features[0].shape[0]
        if self.target_length > fbank_features.shape[0]:
            m = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank_features = m(fbank_features)

        fbank_features = fbank_features.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1) # time, freq
        fbank_features = (fbank_features - self.mean) / (self.std * 2)

        return{
            "audio":fbank_features.unsqueeze(1),
            "label":torch.Tensor(batch["target"]),
        }



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