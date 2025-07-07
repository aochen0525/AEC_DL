import sys
sys.path.append('C:/Users/aochen/Desktop/Acoustic_DL/AEC_DL')
import torch
import torchaudio
from models.transformer_aec import TransformerAEC
import argparse

def real_time_inference(model_path, input_audio_path, output_path, segment_length=20000):
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
    
    # Process in chunks
    num_samples = audio.shape[1]
    enhanced_chunks = []
    for start in range(0, num_samples, segment_length):
        end = min(start + segment_length, num_samples)
        chunk = audio[:, start:end]
        # Pad if last chunk is shorter
        if chunk.shape[1] < segment_length:
            pad = segment_length - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        with torch.no_grad():
            # Model expects [batch, seq_len, channels]
            chunk_input = chunk.unsqueeze(0).transpose(1, 2)  # [1, segment_length, 1]
            enhanced_chunk = model(chunk_input)  # Output: [1, segment_length, 1]
            
            # Remove batch dimension and convert back to [channels, samples]
            enhanced_chunk = enhanced_chunk.squeeze(0)  # [segment_length, 1]
            enhanced_chunk = enhanced_chunk.transpose(0, 1)  # [1, segment_length]
            
            # Remove padding if any
            enhanced_chunk = enhanced_chunk[:, :end-start]
            enhanced_chunks.append(enhanced_chunk)
    
    enhanced_audio = torch.cat(enhanced_chunks, dim=1)

    # Save result
    torchaudio.save(output_path, enhanced_audio.cpu(), sr)
    print(f"Enhanced audio saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--segment_length', type=int, default=20000)
    args = parser.parse_args()
    
    real_time_inference(args.model_path, args.input_path, args.output_path, args.segment_length)