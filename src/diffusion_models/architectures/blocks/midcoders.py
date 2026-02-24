import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CBAM,
    MBConv,
    ResidualLayer,
    ResidualLayer4D,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Midcoder1D(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )

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


class CBAMMidcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels, cond_dim) for _ in range(num_residual_layers)]
        )
        self.cbam_blocks = nn.ModuleList(
            [
                CBAM(channels, cbam_reduction_ratio, cbam_kernel_size)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks and CBAM blocks: (bs, c, L) -> (bs, c, L)
        for res_block, cbam_block in zip(self.res_blocks, self.cbam_blocks):
            x = res_block(x, cond)
            x = cbam_block(x)

        return x


class MBConvMidcoder(Midcoder1D):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
    ):
        super().__init__(channels, num_residual_layers, cond_dim)
        self.mbconv_blocks = nn.ModuleList(
            [
                MBConv(channels, channels, cond_dim, expansion_factor, kernel_size, 1)
                for _ in range(num_mbconv_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Pass through MBConv blocks
        for mbconv in self.mbconv_blocks:
            x = mbconv(x, cond_embed)

        return x


class Midcoder4D(Midcoder1D):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__(channels, num_residual_layers, cond_dim)
        self.res_blocks = nn.ModuleList(
            [ResidualLayer4D(channels, cond_dim) for _ in range(num_residual_layers)]
        )


class TFiLMMidcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.tfilm = TFiLM(
            num_blocks=num_tfilm_blocks,
            channels=channels,
            rnn_hidden=hidden_size_rnn,
            rnn_layers=num_layers_rnn,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Apply TFiLM: (bs, c, L) -> (bs, c, L)
        x = self.tfilm(x, cond=cond_embed)

        return x


class TransFiLMMidcoder(TFiLMMidcoder):
    """
    Midcoder that uses self-attention via a transformer across the time domain for 1D signals. Used to capture long-range time dependencies.
    """

    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        number_transformer_heads: int,
        num_transformer_layers: int,
        ffn_dim_multiplier: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn=channels
            * 2,  # Use dummy value since we won't use the RNN in this midcoder
            num_layers_rnn=1,
        )
        self.tfilm = TFiLMTransformer(
            cond_dim,
            num_tfilm_blocks,
            channels,
            number_transformer_heads,
            num_transformer_layers,
            ffn_dim_multiplier,
        )
