import torch
import torch.nn as nn
import torch.nn.functional as F

class AECLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def stft_loss(self, pred, target, n_fft=512):
        pred_stft = torch.stft(pred.squeeze(-1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        target_stft = torch.stft(target.squeeze(-1), n_fft=n_fft, hop_length=n_fft//4, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        return F.l1_loss(pred_mag, target_mag)
    
    def forward(self, pred, target):
        time_loss = F.l1_loss(pred, target)
        freq_loss = self.stft_loss(pred, target)
        return self.alpha * time_loss + self.beta * freq_loss

