from typing import List

import torch
import torch.nn as nn

from ..dynamics.base import ConditionalVectorField
from .blocks.base import Conditioner, Decoder, Encoder, Midcoder


class MNISTUNet(ConditionalVectorField):
    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        cond_dim: int,
        num_classes: int,
        input_channel: int = 1,
    ):
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Initialize conditioner
        self.conditioner = Conditioner(
            num_classes=num_classes,  # e.g., 3 for amplitude classes
            t_dim=64,  # time embedding dimension
            y_dim=16,  # class embedding dimension
            cond_dim=cond_dim,  # final conditioning dimension
        )

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, cond_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, cond_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, cond_dim)

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
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, 32, 32)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(x, cond)  # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, cond)
        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()  # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, cond)  # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x)  # (bs, 1, 32, 32)

        return x
