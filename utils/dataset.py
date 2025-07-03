import sys
sys.path.append('C:/Users/aochen/Desktop/AudioDL_project')

import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np

class AECChallengeDataset(Dataset):
    def __init__(self, data_dir, segment_length=16000, sample_rate=16000, mode='train'):
        """
        AEC Challenge Dataset for Microsoft AEC-Challenge repo
        
        Args:
            data_dir: Path to 'synthetic' or 'real' folder from AEC-Challenge
            segment_length: Audio segment length in samples
            sample_rate: Target sample rate (16kHz)
            mode: 'train' or 'test'
        """
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.mode = mode
        
        # AEC Challenge directory structure
        self.nearend_mic_dir = os.path.join(data_dir, 'nearend_mic_signal')
        self.nearend_speech_dir = os.path.join(data_dir, 'nearend_speech') 
        self.farend_speech_dir = os.path.join(data_dir, 'farend_speech')
        
        # Verify directories exist
        if not os.path.exists(self.nearend_mic_dir):
            raise ValueError(f"Directory not found: {self.nearend_mic_dir}")
        if not os.path.exists(self.nearend_speech_dir):
            raise ValueError(f"Directory not found: {self.nearend_speech_dir}")
            
        # Load file lists - AEC Challenge files have .wav extensions
        self.audio_files = sorted([f for f in os.listdir(self.nearend_mic_dir) 
                                 if f.endswith('.wav') and not f.startswith('.') and os.path.isfile(os.path.join(self.nearend_mic_dir, f))])
          
    def __len__(self):
        return len(self.audio_files)
    
    def _extract_file_id(self, filename):
        """Extract file ID from filename like 'nearend_mic_fileid_0.wav' -> 'fileid_0'"""
        # Remove extension first
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[-2:])  # Get last two parts: 'fileid_0'
        return name_without_ext
    
    def _construct_filename(self, base_name, file_id):
        """Construct filename for different signal types"""
        return f"{base_name}_{file_id}.wav"
    
    def __getitem__(self, idx):
        mic_filename = self.audio_files[idx]
        
        # Extract file ID from mic filename
        file_id = self._extract_file_id(mic_filename)
        
        # Load nearend microphone signal (input - contains echo)
        mic_path = os.path.join(self.nearend_mic_dir, mic_filename)
        try:
            mic_signal, sr = torchaudio.load(mic_path)
        except Exception as e:
            print(f"Error loading mic signal {mic_path}: {e}")
            # Return dummy data if loading fails
            dummy_signal = torch.zeros(1, self.segment_length)
            return (dummy_signal.transpose(0, 1), dummy_signal.transpose(0, 1), dummy_signal.transpose(0, 1))
        
        # Construct clean speech filename
        clean_filename = self._construct_filename('nearend_speech', file_id)
        clean_path = os.path.join(self.nearend_speech_dir, clean_filename)
        
        # Load clean nearend speech (target - without echo)
        try:
            clean_signal, _ = torchaudio.load(clean_path)
        except Exception as e:
            print(f"Warning: Could not load clean signal {clean_path}: {e}")
            print(f"Using mic signal as fallback")
            clean_signal = mic_signal.clone()
        
        # Construct farend speech filename
        farend_filename = self._construct_filename('farend_speech', file_id)
        farend_path = os.path.join(self.farend_speech_dir, farend_filename)

        # Load farend speech (reference signal) - optional
        farend_signal = None
        if os.path.exists(farend_path):
            try:
                farend_signal, _ = torchaudio.load(farend_path)
            except Exception as e:
                print(f"Warning: Could not load farend signal {farend_path}: {e}")
                farend_signal = None
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            mic_signal = resampler(mic_signal)
            clean_signal = resampler(clean_signal)
            if farend_signal is not None:
                farend_signal = resampler(farend_signal)
        
        # Use mono channel
        if mic_signal.shape[0] > 1:
            mic_signal = mic_signal[0:1, :]
        if clean_signal.shape[0] > 1:
            clean_signal = clean_signal[0:1, :]
        if farend_signal is not None and farend_signal.shape[0] > 1:
            farend_signal = farend_signal[0:1, :]
        
        # Ensure signals have same length
        min_length = min(mic_signal.shape[1], clean_signal.shape[1])
        if farend_signal is not None:
            min_length = min(min_length, farend_signal.shape[1])
        
        mic_signal = mic_signal[:, :min_length]
        clean_signal = clean_signal[:, :min_length]
        if farend_signal is not None:
            farend_signal = farend_signal[:, :min_length]
        
        # Segment to fixed length
        if min_length > self.segment_length:
            if self.mode == 'train':
                start = np.random.randint(0, min_length - self.segment_length)
            else:
                start = (min_length - self.segment_length) // 2
            
            mic_signal = mic_signal[:, start:start + self.segment_length]
            clean_signal = clean_signal[:, start:start + self.segment_length]
            if farend_signal is not None:
                farend_signal = farend_signal[:, start:start + self.segment_length]
        else:
            # Pad if too short
            pad_length = self.segment_length - min_length
            mic_signal = torch.nn.functional.pad(mic_signal, (0, pad_length))
            clean_signal = torch.nn.functional.pad(clean_signal, (0, pad_length))
            if farend_signal is not None:
                farend_signal = torch.nn.functional.pad(farend_signal, (0, pad_length))

        if farend_signal is None:
            farend_signal = torch.zeros_like(mic_signal)  

        # Return format: (mic_signal, farend_signal, clean_signal)
        # Keep as [channels, seq_len] - DataLoader will add batch dimension to make [batch, channels, seq_len]
        return (mic_signal, farend_signal, clean_signal)
