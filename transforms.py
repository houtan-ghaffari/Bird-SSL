from dataclasses import dataclass
import numpy as np
from transformers.audio_utils import spectrogram, window_function, mel_filter_bank
from omegaconf import DictConfig
import torch 

class Transform:
    def __init__(self, 
                 mel_params: DictConfig,
                 spectrogram_params: DictConfig,
                 window_params: DictConfig,             
                 target_length: int
        ):
        self.sampling_rate = mel_params.sampling_rate  
        self.target_length = target_length 
        self.spectrogram_params = spectrogram_params  
        self.mel_filter_params = mel_params    

        self.mel_filters = self._init_mel_filters()
        self.window = window_function(
            window_length=self.spectrogram_params.frame_length, 
            name=window_params.type, 
            periodic=window_params.periodic)

        self.target_length = target_length        

    def cyclic_rolling_start(self, waveform):
        idx = np.random.randint(0, len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        volume_mag = np.random.beta(10,10) + 0.5
        waveform = rolled_waveform * volume_mag
        waveform = waveform - waveform.mean()
        return waveform

    def _init_mel_filters(self):
        mel_filters = mel_filter_bank(**self.mel_filter_params)
        return np.pad(mel_filters, ((0, 1), (0, 0)))
    	
    def __call__(self, batch):

        input = []
        for audio in batch["audio"]:
            wav = self.cyclic_rolling_start(audio["array"])
            wav = spectrogram(
                wav,
                mel_filters=self.mel_filters, 
                window=self.window, 
                **self.spectrogram_params).T
            wav = torch.from_numpy(wav)
            # Apply cyclic rolling start
            
            # Apply spectrogram

            n_frames = wav.shape[0]
            p = self.target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                wav = m(wav)
            elif p < 0:
                wav = wav[0:self.target_length, :]
            
            input.append(wav.unsqueeze(0))
    
        return{
            "audio":input,
            "label":torch.Tensor(batch["target"]),
        }

