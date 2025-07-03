import sys
sys.path.append('C:/Users/aochen/Desktop/AudioDL_project')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from models.transformer_aec import TransformerAEC, DualPathTransformerAEC
from utils.dataset import AECChallengeDataset
from utils.loss_functions import AECLoss
import argparse

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (mic_signal, farend_signal, clean_signal) in enumerate(dataloader):
        # Debug: Check original shapes from dataloader
        print(f"Debug - Original shapes from dataloader: mic={mic_signal.shape}, clean={clean_signal.shape}")
        
        # Dataset returns [batch, channels, seq_len], we need [batch, seq_len, channels] for transformer
        mic_signal = mic_signal.transpose(1, 2).to(device)  # [batch, seq_len, channels]
        clean_signal = clean_signal.transpose(1, 2).to(device)  # [batch, seq_len, channels]
        
        print(f"Debug - Final input shapes: mic={mic_signal.shape}, clean={clean_signal.shape}")
        
        optimizer.zero_grad()
        
        # Forward pass
        enhanced_signal = model(mic_signal)
        print(f"Debug - Output shape: {enhanced_signal.shape}")
        loss = criterion(enhanced_signal, clean_signal)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
            print(f'Input shape: {mic_signal.shape}, Output shape: {enhanced_signal.shape}')
        
        # Stop after first batch for debugging
        if batch_idx == 0:
            print("Stopping after first batch for dimension debugging")
            break
    
    scheduler.step()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='transformer', choices=['transformer', 'dual_path'])
    parser.add_argument('--hidden_dim', type=int, default=256)  # Reduced from 512 to avoid division issues
    parser.add_argument('--num_layers', type=int, default=4)  # Reduced from 6 for faster testing
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    try:
        if args.model_type == 'transformer':
            model = TransformerAEC(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        else:
            model = DualPathTransformerAEC(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        
        model = model.to(device)
        print(f"✓ Model created successfully")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    # Loss and optimizer
    criterion = AECLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Dataset and dataloader
    train_data_dir = 'AEC-Challenge/datasets/synthetic'
    train_dataset = AECChallengeDataset(train_data_dir, segment_length=8000)  # Smaller segment for debugging
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)  # Small batch, no workers for debugging
    
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with one batch first
    print("Testing with one batch...")
    sample_batch = next(iter(train_loader))
    print(f"Sample batch shapes: {[x.shape for x in sample_batch]}")
    
    # Create checkpoints directory
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop - only run one epoch for debugging
    for epoch in range(1):  # Changed from args.epochs to 1 for debugging
        print(f"Starting epoch {epoch}")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoints/model_epoch_{epoch}.pth')
            
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
