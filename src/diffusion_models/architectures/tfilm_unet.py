from typing import List

import torch
import torch.nn as nn

from .blocks.tfilm import TFiLMEncoder, TFiLMMidcoder, TFiLMDecoder
from .blocks.base import Conditioner

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
    
class TFiLMUNetTransformer(nn.Module):
    """
    UNet with TFiLM conditioning for 1D signals (using Transformer-based TFiLM blocks)
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
            encoders.append(TFiLMEncoder(curr_c, next_c, num_residual_layers, num_tfilm_blocks, cond_dim, use_transformer=True))
            decoders.append(TFiLMDecoder(next_c, curr_c, num_residual_layers, num_tfilm_blocks, cond_dim, use_transformer=True))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMidcoder(channels[-1], num_residual_layers, num_tfilm_blocks, cond_dim, use_transformer=True)
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