import sys
sys.path.append('C:/Users/aochen/Desktop/Acoustic_DL/AEC-Challenge')

import torch
from models.transformer_aec import TransformerAEC
from utils.dataset import AECChallengeDataset
from torch.utils.data import DataLoader

def test_dimensions():
    print("Testing dimension compatibility...")
    
    # Create dataset and dataloader
    data_dir = 'C:/Users/aochen/Desktop/Acoustic_DL/AEC-Challenge/datasets/synthetic'
    dataset = AECChallengeDataset(data_dir, segment_length=1000)  # Small segment for testing
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create model
    model = TransformerAEC(input_dim=1, hidden_dim=256, num_layers=2)  # Smaller model for testing
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get one batch
    batch = next(iter(dataloader))
    mic_signal, farend_signal, clean_signal = batch
    
    print(f"Original shapes from dataloader:")
    print(f"  Mic: {mic_signal.shape}")
    print(f"  Farend: {farend_signal.shape}")
    print(f"  Clean: {clean_signal.shape}")
    
    # Convert to correct format for transformer
    mic_signal = mic_signal.transpose(1, 2)  # [batch, seq_len, channels]
    clean_signal = clean_signal.transpose(1, 2)
    
    print(f"Shapes after transpose:")
    print(f"  Mic: {mic_signal.shape}")
    print(f"  Clean: {clean_signal.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model(mic_signal)
        print(f"Output shape: {output.shape}")
        
    print("✓ Dimension test passed!")

if __name__ == "__main__":
    test_dimensions()
