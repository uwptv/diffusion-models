import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import ResidualLayer, ResidualLayer1D
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels, cond_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, cond)

        return x


class Midcoder1D(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer1D(channels, cond_dim) for _ in range(num_residual_layers)]
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


class MidcoderTransformer1D(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        num_transformer_layers: int,
        cond_dim: int,
        nhead: int = 8,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer1D(channels, cond_dim) for _ in range(num_residual_layers)]
        )

        encoderLayer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=4 * channels,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoderLayer, num_layers=num_transformer_layers
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Residual blocks first: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Reshape for transformer: (bs, c, L) -> (bs, L, c)
        x = x.transpose(1, 2)

        # Self-attention: (bs, L, c) -> (bs, L, c)
        x = self.transformer(x)

        # Reshape back: (bs, L, c) -> (bs, c, L)
        x = x.transpose(1, 2)

        return x


class TFiLMMidcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        use_transformer: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.activation = nn.ReLU()
        if use_transformer:
            self.tfilm = TFiLMTransformer(
                num_blocks=num_tfilm_blocks,
                channels=channels,
                num_heads=8,
                num_layers=6,
            )
        else:
            self.tfilm = TFiLM(
                num_blocks=num_tfilm_blocks, channels=channels, rnn_hidden=256
            )

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
