import torch.nn as nn

from diffusion_models.architectures.blocks.base import InitialConvolution
from diffusion_models.architectures.blocks.decoders import Decoder1D
from diffusion_models.architectures.blocks.encoders import Encoder1D
from diffusion_models.architectures.blocks.midcoders import Midcoder1D
from diffusion_models.architectures.unet import UNet


class StandardUNet(UNet):
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
        input_channels: int,
        initial_channels: int,
        levels: int,
        num_residual_layers: int,
        num_classes: int,
        cond_dim: int,
    ):
        super().__init__(cond_dim, num_classes)

        self.init_conv = InitialConvolution(
            input_channels, initial_channels, cond_dim=cond_dim, use_1d=True
        )

        channels = [initial_channels]
        for _ in range(levels):
            # Double channels at each level
            channels.append(channels[-1] * 2)

        # Encoders and Decoders (use cond_dim for both t_embed_dim and y_embed_dim)
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder1D(curr_c, next_c, num_residual_layers, cond_dim))
            decoders.append(Decoder1D(next_c, curr_c, num_residual_layers, cond_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder1D(channels[-1], num_residual_layers, cond_dim)
        self.final_conv = nn.Conv1d(
            channels[0], input_channels, kernel_size=3, padding=1
        )
