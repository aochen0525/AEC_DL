import sys
sys.path.append('C:/Users/aochen/Desktop/AudioDL_project')
import torch
import torchaudio
from models.transformer_aec import TransformerAEC
import argparse

def real_time_inference(model_path, input_audio_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = TransformerAEC(input_dim=1, use_causal_mask=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Load audio
    audio, sr = torchaudio.load(input_audio_path)
    audio = audio.to(device)
    
    # Process audio
    with torch.no_grad():
        enhanced_audio = model(audio.unsqueeze(0).transpose(1, 2))
        enhanced_audio = enhanced_audio.squeeze().transpose(0, 1)
    
    # Save result
    torchaudio.save(output_path, enhanced_audio.cpu(), sr)
    print(f"Enhanced audio saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    
    real_time_inference(args.model_path, args.input_path, args.output_path)