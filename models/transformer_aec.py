import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Ensure divisible by 3
        conv_out_channels = out_channels // 3
        remaining_channels = out_channels - 2 * conv_out_channels
        
        self.conv1 = nn.Conv1d(in_channels, conv_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, conv_out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, remaining_channels, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.transpose(1, 2)  # back to (batch, seq_len, channels)
        return self.norm(x)

class TransformerAEC(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=6, hidden_dim=512, 
                 dropout=0.1, use_causal_mask=True):
        super().__init__()
        
        # Multi-scale feature extraction
        self.feature_extractor = MultiScaleConv1d(input_dim, hidden_dim//2)
        self.input_proj = nn.Linear(hidden_dim//2, hidden_dim)
        
        # Positional encoding for temporal modeling
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Enhanced transformer with residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Causal masking for real-time processing
        self.use_causal_mask = use_causal_mask
        
        # Multi-head output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, input_dim)
        )
        
        # Residual connection weight
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def _generate_causal_mask(self, seq_len, device):
        """Generate causal mask for real-time processing"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, x, reference_signal=None):
        """
        x: microphone signal (batch, seq_len, input_dim)
        reference_signal: loudspeaker signal (batch, seq_len, input_dim) - optional
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale feature extraction
        features = self.feature_extractor(x)
        features = self.input_proj(features)
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Generate causal mask for real-time processing
        mask = None
        if self.use_causal_mask:
            mask = self._generate_causal_mask(seq_len, x.device)
        
        # Transformer processing
        enhanced = self.transformer(features, mask=mask)
        
        # Output projection
        output = self.output_proj(enhanced)
        
        # Residual connection with learnable weight
        output = self.alpha * output + (1 - self.alpha) * x
        
        return output

class DualPathTransformerAEC(nn.Module):
    """Dual-path processing for better echo cancellation"""
    def __init__(self, input_dim, chunk_size=64, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.intra_chunk_transformer = TransformerAEC(input_dim, **kwargs)
        self.inter_chunk_transformer = TransformerAEC(input_dim, **kwargs)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Reshape for dual-path processing
        chunks = x.view(batch_size, -1, self.chunk_size, input_dim)
        num_chunks = chunks.shape[1]
        
        # Intra-chunk processing
        chunks_flat = chunks.view(-1, self.chunk_size, input_dim)
        processed_chunks = self.intra_chunk_transformer(chunks_flat)
        processed_chunks = processed_chunks.view(batch_size, num_chunks, self.chunk_size, input_dim)
        
        # Inter-chunk processing
        inter_input = processed_chunks.transpose(1, 2).contiguous().view(batch_size * self.chunk_size, num_chunks, input_dim)
        inter_output = self.inter_chunk_transformer(inter_input)
        inter_output = inter_output.view(batch_size, self.chunk_size, num_chunks, input_dim).transpose(1, 2)
        
        # Reshape back
        output = inter_output.contiguous().view(batch_size, -1, input_dim)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :-pad_len]
        
        return output