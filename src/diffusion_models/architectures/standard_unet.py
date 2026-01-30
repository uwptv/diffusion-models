from typing import List

import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import Conditioner, InitialConvolution
from diffusion_models.architectures.blocks.decoders import Decoder1D
from diffusion_models.architectures.blocks.encoders import Encoder1D
from diffusion_models.architectures.blocks.midcoders import Midcoder1D
from diffusion_models.dynamics.base import ConditionalVectorField


class StandardUNet(ConditionalVectorField):
    """
    1D UNet for conditional (sine) wave generation.
    Uses an initial convolution to extend the channel dimension, followed by a series of encoder
    blocks, a midcoder block, decoders blocks, and a final convolution to reduce the channel dimension back to 1.
    Each encoder block applies residual layers and 1D convolution downsampling.
    The midcoder block only applies residual layers.
    Each decoder block applies upsampling, 1D convolution to reduce channels, and residual layers.
    Lastly, the channels are reduced back to 1 using a final convolution.
    Conditioning is done via a unified conditioning vector obtained from time and class embeddings which is applied to all encoder, midcoder, and decoder blocks.
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        cond_dim: int,
        num_classes: int,
        input_channels: int = 1,
    ):
        super().__init__()

        self.init_conv = InitialConvolution(
            input_channels, channels[0], cond_dim=cond_dim, use_1d=True
        )

        # Replace separate embedders with Conditioner
        self.conditioner = Conditioner(
            num_classes=num_classes,  # e.g., 3 for amplitude classes
            t_dim=64,  # time embedding dimension
            y_dim=16,  # class embedding dimension
            cond_dim=cond_dim,  # final conditioning dimension
        )

        # Encoders and Decoders (use cond_dim for both t_embed_dim and y_embed_dim)
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
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

        x = self.init_conv(x, cond)
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
