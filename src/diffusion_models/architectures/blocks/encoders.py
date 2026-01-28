import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import ResidualLayer, get_activation
from diffusion_models.architectures.blocks.one_d_base import (
    ResidualLayer1D,
    SeperableConv1D,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_in, cond_dim) for _ in range(num_residual_layers)]
        )
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, cond)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Encoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer1D(channels_in, cond_dim) for _ in range(num_residual_layers)]
        )
        self.downsample = nn.Conv1d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )

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


class TFiLMEncoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        activation: str = "relu",
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        use_transformer: bool = False,
        use_seperable_conv: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels_in, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
        if use_seperable_conv:
            self.downsample = SeperableConv1D(
                channels_in=channels_in,
                channels_out=channels_out,
                filters_per_channel=4,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            )
        else:
            self.downsample = nn.Conv1d(
                channels_in,
                channels_out,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
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
