import sys
sys.path.append('C:/Users/aochen/Desktop/Acoustic_DL/AEC_DL')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from models.transformer_aec import TransformerAEC, DualPathTransformerAEC
from utils.dataset import AECChallengeDataset
from utils.loss_functions import AECLoss
import argparse
import os
import time
from torch.amp import autocast, GradScaler

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Gradient accumulation for effective larger batch size
    accumulation_steps = 4  # Accumulate gradients over 4 mini-batches
    
    for batch_idx, (mic_signal, farend_signal, clean_signal) in enumerate(dataloader):
        # Dataset returns [batch, channels, seq_len], we need [batch, seq_len, channels] for transformer
        mic_signal = mic_signal.transpose(1, 2).to(device, non_blocking=True)  # [batch, seq_len, channels]
        clean_signal = clean_signal.transpose(1, 2).to(device, non_blocking=True)  # [batch, seq_len, channels]
        
        # Mixed precision forward pass
        with autocast('cuda'):
            enhanced_signal = model(mic_signal)
            loss = criterion(enhanced_signal, clean_signal)
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Undo the scaling for logging
        
        # Print progress every 10 batches to reduce I/O overhead
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            gpu_memory = f", GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else ""
            print(f'Epoch: {epoch + 1}, Batch: {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item() * accumulation_steps:.6f}, Time: {elapsed:.1f}s{gpu_memory}')
    
    # Handle any remaining gradients
    if (len(dataloader) % accumulation_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    scheduler.step()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (mic_signal, farend_signal, clean_signal) in enumerate(dataloader):
            mic_signal = mic_signal.transpose(1, 2).to(device)
            clean_signal = clean_signal.transpose(1, 2).to(device)
            
            enhanced_signal = model(mic_signal)
            loss = criterion(enhanced_signal, clean_signal)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='transformer', choices=['transformer', 'dual_path'])
    parser.add_argument('--hidden_dim', type=int, default=512)  # Restored to full size
    parser.add_argument('--num_layers', type=int, default=6)  # Restored to full size
    parser.add_argument('--batch_size', type=int, default=2)  # Increased from 4 for better GPU utilization
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'test'], 
                        help='Mode: train, validation, or test')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear GPU cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    if args.use_wandb:
        wandb.init(
            project="acoustic-echo-cancellation",
            config={
                "model_type": args.model_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
            }
        )
    

    # Initialize model
    try:
        if args.model_type == 'transformer':
            model = TransformerAEC(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        else:
            model = DualPathTransformerAEC(input_dim=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        
        model = model.to(device)
        print(f"✓ Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    # Loss and optimizer
    criterion = AECLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Dataset and dataloader
    data_dir = 'C:/Users/aochen/Desktop/Acoustic_DL/AEC-Challenge/datasets/synthetic'
    full_dataset = AECChallengeDataset(data_dir, segment_length=16000)  # Full segment length
    total_len = len(full_dataset)
    val_len = int(0.1 * total_len)
    test_len = int(0.1 * total_len)
    train_len = total_len - val_len - test_len

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
    if args.mode == 'train':
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    elif args.mode == 'validation':
        loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    elif args.mode == 'test':
        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
    #                           num_workers=12, pin_memory=True, persistent_workers=True, 
    #                           prefetch_factor=4)  # Prefetch more batches
    
    # # Validation dataset if requested
    # val_loader = None
    # if args.validate:
    #     # You might want to create a separate validation dataset
    #     val_dataset = AECChallengeDataset(train_data_dir, segment_length=16000)  # Use same for now
    #     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
    #                             num_workers=2, pin_memory=True)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(loader)}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\n{'='*50}")
            print(f"Starting epoch {epoch + 1}/{args.epochs}")
            print(f"{'='*50}")
            
            # Training
            avg_train_loss = train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, scaler)
            print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}')
            
            # Validation
            if args.mode == 'validation':
                avg_val_loss = validate_epoch(model, loader, criterion, device)
                print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.6f}')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, 'checkpoints/best_model.pth')
                    print(f'✓ New best model saved with validation loss: {best_val_loss:.6f}')
                        # Validation
            elif args.mode == 'test':
                avg_test_loss = validate_epoch(model, loader, criterion, device)
                print(f'Test Loss: {avg_test_loss:.6f}')
            
            # Log to wandb
            if args.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                if args.mode == 'validation':
                    log_dict["val_loss"] = avg_val_loss
                wandb.log(log_dict)
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                }, f'checkpoints/model_epoch_{epoch}.pth')
                print(f'✓ Checkpoint saved: model_epoch_{epoch}.pth')
            
    except KeyboardInterrupt:
        print(f"\n✗ Training interrupted by user at epoch {epoch + 1}")
        print("Saving checkpoint before exit...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
        }, f'checkpoints/model_interrupted_epoch_{epoch}.pth')
        print("✓ Checkpoint saved successfully")
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        
    finally:
        if args.use_wandb:
            wandb.finish()
        print("Training session ended.")
            
if __name__ == '__main__':
    main()
