import torch.nn as nn

from diffusion_models.architectures.blocks.base import InitialConvolution
from diffusion_models.architectures.blocks.decoders import MBConvDecoder
from diffusion_models.architectures.blocks.encoders import MBConvEncoder
from diffusion_models.architectures.blocks.midcoders import MBConvMidcoder
from diffusion_models.architectures.unet import UNet


class MBConvUNet(UNet):
    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        levels: int,
        upsampling_method: str,
        num_residual_layers: int,
        num_classes: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
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
            encoders.append(
                MBConvEncoder(
                    curr_c,
                    next_c,
                    num_residual_layers,
                    cond_dim,
                    num_mbconv_layers,
                    expansion_factor,
                    kernel_size,
                )
            )
            decoders.append(
                MBConvDecoder(
                    next_c,
                    curr_c,
                    upsampling_method,
                    num_residual_layers,
                    cond_dim,
                    num_mbconv_layers,
                    expansion_factor,
                    kernel_size,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = MBConvMidcoder(
            channels[-1],
            num_residual_layers,
            cond_dim,
            num_mbconv_layers,
            expansion_factor,
            kernel_size,
        )
        self.final_conv = nn.Conv1d(
            channels[0], input_channels, kernel_size=3, padding=1
        )
