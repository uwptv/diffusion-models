import math
from typing import List

import torch
from torch import nn

from differential_equations import ConditionalVectorField

class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim)
    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
 
        assert dim % 2 == 0
        self.half_dim = dim // 2
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
 
        t = t.view(-1, 1)
        # (B, 1)
 
        # Compute frequencies: [1, 10000^(2i/d)]
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.half_dim, dtype=torch.float32)
            / self.half_dim
        ).to(t.device)
        # (half_dim,)
 
        angles = t * freqs * 2 * math.pi
        # (B, half_dim)
 
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # (B, dim)
 
        return emb

class Conditioner(nn.Module):
    def __init__(self, num_classes: int, t_dim: int, y_dim: int, cond_dim: int) -> None:
        super().__init__()
 
        self.t_embedder = SinusoidalTimeEmbedding(t_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim)
 
        self.mlp = nn.Sequential(
            nn.Linear(t_dim + y_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
 
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        # y: (B,)
 
        t_embed = self.t_embedder(t) # (B, t_dim)
        y_embed = self.y_embedder(y) # (B, y_dim)
        cond = torch.cat([t_embed, y_embed], dim=1) # (B, t_dim + y_dim)
        cond = self.mlp(cond) # (B, cond_dim)
 
        return cond
    
class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone() # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x) # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1) # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x) # (bs, c, h, w)

        # Add back residual
        x = x + res # (bs, c, h, w)

        return x
        
class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x

class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)
            
        return x
        
class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w) 
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x

class MNISTUNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int): 
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]), nn.SiLU())

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
            
        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop() # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x
    
class ResidualLayer1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int,):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, cond_dim) -> (bs, channels)
        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, channels)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        """
        res = x.clone() # (bs, c, L)

        # Initial conv block
        x = self.block1(x) # (bs, c, L)

        # Add conditioning embedding
        cond = self.cond_adapter(cond).unsqueeze(-1) # (bs, c, 1)
        x = x + cond

        # Second conv block
        x = self.block2(x) # (bs, c, L)

        # Add back residual
        x = x + res # (bs, c, L)

        return x

class Encoder1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_in, cond_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        return x

class Midcoder1D(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels, cond_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)
            
        return x

class Decoder1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_out, cond_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x

class StandardUNet(ConditionalVectorField):
    """
    1D UNet for conditional (sine) wave generation
    """
    def __init__(self, channels: List[int], num_residual_layers: int, 
                 cond_dim: int, num_classes: int, input_channels: int = 1): 
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU()
        )

        # Replace separate embedders with Conditioner
        self.conditioner = Conditioner(
            num_classes=num_classes,  # e.g., 3 for amplitude classes
            t_dim=64,               # time embedding dimension
            y_dim=16,               # class embedding dimension
            cond_dim=cond_dim        # final conditioning dimension
        )

        # Encoders and Decoders (use cond_dim for both t_embed_dim and y_embed_dim)
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder1D(curr_c, next_c, num_residual_layers, cond_dim))
            decoders.append(Decoder1D(next_c, curr_c, num_residual_layers, cond_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder1D(channels[-1], num_residual_layers, cond_dim)
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, L)
        - t: (bs, 1, 1) -> will be squeezed to (bs,)
        - y: (bs,) amplitude class labels
        Returns:
        - u_t^theta(x|y): (bs, 1, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)
        
        
        x = self.init_conv(x)
        residuals = []
        
        for encoder in self.encoders:
            x = encoder(x, cond)
            residuals.append(x.clone())

        x = self.midcoder(x, cond)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, cond)

        x = self.final_conv(x)
        return x
    
class TFiLM(nn.Module):
    def __init__(self, num_blocks: int, channels: int, rnn_hidden: int, rnn_layers: int = 1):
        super().__init__()
        self.num_blocks = num_blocks
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn = nn.LSTM(
            input_size=channels,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=True
        )
        self.to_params = nn.Linear(2 * self.rnn_hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T_orig = x.shape
        
        # Calculate padding needed
        remainder = T_orig % self.num_blocks
        if remainder != 0:
            pad_amount = self.num_blocks - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount), mode='replicate')
        else:
            pad_amount = 0
        
        B, C, T = x.shape
        block_len = T // self.num_blocks

        # (B, C, T) -> (B, num_blocks, C, block_len)
        blocks = x.view(B, C, self.num_blocks, block_len).permute(0, 2, 1, 3)

        # Max-pool over time within each block: (B, num_blocks, C)
        pooled = blocks.max(dim=-1).values

        # RNN over blocks (sequence length = num_blocks)
        rnn_out, _ = self.rnn(pooled)  # (B, num_blocks, 2 * hidden)

        # Affine params per block/channel
        params = self.to_params(rnn_out)  # (B, num_blocks, 2*C)
        scale, shift = params.chunk(2, dim=-1)    # each: (B, num_blocks, C)

        # Apply affine to each block
        scale = scale.unsqueeze(-1)  # (B, num_blocks, C, 1)
        shift = shift.unsqueeze(-1)  # (B, num_blocks, C, 1)
        mod_blocks = scale * blocks + shift      # (B, num_blocks, C, block_len)

        # Reassemble to (B, C, T)
        out = mod_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)
        
        # Remove padding if any was added
        if pad_amount > 0:
            out = out[:, :, :T_orig]
        
        return out
    
class TFiLMEncoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, num_tfilm_blocks: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_in, cond_dim=cond_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU()
        self.tfilm = TFiLM(num_blocks=num_tfilm_blocks, channels=channels_out, rnn_hidden=256)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        # Apply activation: (bs, c_out, L // 2) -> (bs, c_out, L // 2)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_out, L // 2) -> (bs, c_out, L // 2)
        x = self.tfilm(x)

        return x
    
class TFiLMDecoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, num_tfilm_blocks: int, cond_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_out, cond_dim=cond_dim) for _ in range(num_residual_layers)
        ])
        self.activation = nn.ReLU()
        self.tfilm = TFiLM(num_blocks=num_tfilm_blocks, channels=channels_out, rnn_hidden=256)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Apply activation: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.tfilm(x)

        return x
    
class TFiLMMidcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, num_tfilm_blocks: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels, cond_dim=cond_dim) for _ in range(num_residual_layers)
        ])
        self.activation = nn.ReLU()
        self.tfilm = TFiLM(num_blocks=num_tfilm_blocks, channels=channels, rnn_hidden=256)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Apply activation: (bs, c, L) -> (bs, c, L)
        x = self.activation(x)

        # Apply TFiLM: (bs, c, L) -> (bs, c, L)
        x = self.tfilm(x)

        return x
    
class TFiLMUNet(nn.Module):
    """
    UNet with TFiLM conditioning for 1D signals
    """
    def __init__(self, channels: List[int], num_residual_layers: int, num_tfilm_blocks: int, num_classes: int, cond_dim: int, input_channels: int = 3): 
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU()
        )
        self.conditioner = Conditioner(
            num_classes=num_classes,  # e.g., 3 for amplitude classes
            t_dim=64,               # time embedding dimension
            y_dim=16,               # class embedding dimension
            cond_dim=cond_dim        # final conditioning dimension
        )

        # Encoders and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(TFiLMEncoder(curr_c, next_c, num_residual_layers, num_tfilm_blocks, cond_dim))
            decoders.append(TFiLMDecoder(next_c, curr_c, num_residual_layers, num_tfilm_blocks, cond_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMidcoder(channels[-1], num_residual_layers, num_tfilm_blocks, cond_dim)
        self.final_conv = nn.Conv1d(channels[0], input_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, L)
        - t: (bs, 1, 1) -> will be squeezed to (bs,)
        - y: (bs, 1) amplitude class labels
        Returns:
        - u_t^theta(x|y): (bs, 1, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)
        
        x = self.init_conv(x)
        residuals = []
        
        for encoder in self.encoders:
            x = encoder(x, cond)
            residuals.append(x.clone())

        x = self.midcoder(x, cond)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, cond)

        x = self.final_conv(x)
        return x

if __name__ == "__main__":
    # Test TUNet
    model = StandardUNet(
        channels = [32, 64, 128],
        num_residual_layers = 2,
        cond_dim = 40,
        num_classes = 10,  # For example, if you have 10 classes
    )
    x = torch.randn(4, 1, 628)  # (bs=4, channels=1, signal_length=628)
    t = torch.randn(4, 1, 1)    # (bs=4, 1, 1)
    y = torch.randint(0, 10, (4, 1))  # (bs=4, 1) with class indices
    out = model(x, t, y)
    print(out.shape)  # Expected output shape: (4, 1, 628)