import torch.nn as nn

from diffusion_models.architectures.blocks.base import AdaGroupNorm
from diffusion_models.architectures.blocks.decoders import CBAMDecoder
from diffusion_models.architectures.blocks.encoders import CBAMEncoder
from diffusion_models.architectures.blocks.midcoders import CBAMMidcoder
from diffusion_models.architectures.unet import UNet


class CBAMUNet(UNet):
    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        upsampling_method: str,
        levels: int,
        num_residual_layers: int,
        num_classes: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
    ):
        super().__init__(cond_dim, num_classes)

        self.init_conv = nn.Conv1d(
            input_channels, initial_channels, kernel_size=3, padding=1
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
                CBAMEncoder(
                    curr_c,
                    next_c,
                    num_residual_layers,
                    cond_dim,
                    cbam_reduction_ratio,
                    cbam_kernel_size,
                )
            )
            decoders.append(
                CBAMDecoder(
                    2 * next_c,
                    curr_c,
                    upsampling_method,
                    num_residual_layers,
                    cond_dim,
                    cbam_reduction_ratio,
                    cbam_kernel_size,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = CBAMMidcoder(
            channels[-1],
            num_residual_layers,
            cond_dim,
            cbam_reduction_ratio,
            cbam_kernel_size,
        )
        self.final_conv = nn.Sequential(
            AdaGroupNorm(num_channels=initial_channels, cond_dim=cond_dim),
            nn.SiLU(),
            nn.Conv1d(initial_channels, input_channels, kernel_size=3, padding=1),
        )
