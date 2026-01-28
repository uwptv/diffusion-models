import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import ResidualLayer, get_activation
from diffusion_models.architectures.blocks.one_d_base import (
    ResidualLayer1D,
    SeperableConv1D,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_out, cond_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - cond: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, cond)

        return x


class Decoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels_out, cond_dim)
                for _ in range(num_residual_layers)
            ]
        )

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


class TFiLMDecoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        activation: str = "relu",
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        conv_output_padding: int = 0,
        use_transpose_conv: bool = False,
        use_transformer: bool = False,
        use_seperable_conv: bool = False,
    ):
        super().__init__()
        if use_transpose_conv:
            self.upsample = nn.ConvTranspose1d(
                channels_in,
                channels_out,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
                output_padding=conv_output_padding,
            )
        else:
            if use_seperable_conv:
                conv = SeperableConv1D(
                    channels_in,
                    channels_out,
                    filters_per_channel=4,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                )
            else:
                conv = nn.Conv1d(
                    channels_in,
                    channels_out,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                )
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                conv,
            )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels_out, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.activation = get_activation(activation)
        if use_transformer:
            self.tfilm = TFiLMTransformer(
                num_blocks=num_tfilm_blocks,
                channels=channels_out,
                num_heads=8,
                num_layers=6,
            )
        else:
            self.tfilm = TFiLM(
                num_blocks=num_tfilm_blocks, channels=channels_out, rnn_hidden=256
            )

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
