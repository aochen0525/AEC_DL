import sys
sys.path.append('C:/Users/aochen/Desktop/Acoustic_DL')
import torch
import torchaudio
from models.transformer_aec import TransformerAEC
import numpy as np
import os
from pesq import pesq
from pystoi import stoi


def evaluate_model(model_path, test_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = TransformerAEC(input_dim=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation metrics
    pesq_scores = []
    stoi_scores = []
    
    with torch.no_grad():
        for audio_file in os.listdir(test_dir):
            if audio_file.endswith('.wav'):
                # Load test audio
                audio_path = os.path.join(test_dir, audio_file)
                mic_signal, sr = torchaudio.load(audio_path)
                
                # Process with model
                enhanced = model(mic_signal.unsqueeze(0).transpose(1, 2))
                enhanced = enhanced.squeeze().cpu().numpy()
                
                # Calculate metrics (you'll need reference clean audio)
                # pesq_score = pesq(sr, clean_audio, enhanced, 'wb')
                # stoi_score = stoi(clean_audio, enhanced, sr)
    
    return np.mean(pesq_scores), np.mean(stoi_scores)