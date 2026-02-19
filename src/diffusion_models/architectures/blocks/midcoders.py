import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    AdaGroupNorm,
    ConditionalCBAM,
    MBConv,
    ResidualLayer,
    ResidualLayer4D,
)
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
            [
                ResidualLayer(channels, cond_dim, use_1d=True, num_groups=channels)
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


class MidcoderTransformer1D(nn.Module):
    """
    Midcoder with that uses self-attention via a transformer across the time domain for 1D signals. Used to capture long-range time dependencies.
    """

    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        num_transformer_layers: int,
        cond_dim: int,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )

        self.transformer_layers = nn.ModuleList(
            [
                _MidcoderTransformerLayer(
                    d_model=channels,
                    num_heads=nhead,
                    dim_feedforward=4 * channels,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
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
        for layer in self.transformer_layers:
            x = layer(x)

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
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
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
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Apply TFiLM: (bs, c, L) -> (bs, c, L)
        x = self.tfilm(x, cond=cond_embed)

        return x


class CBAMMidcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels, cond_dim) for _ in range(num_residual_layers)]
        )
        self.cbam_blocks = nn.ModuleList(
            [ConditionalCBAM(channels, cond_dim) for _ in range(num_residual_layers)]
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
            x = cbam_block(x, cond)

        return x


class TFiLMMBConvMidcoder(TFiLMMidcoder):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        use_transformer: bool = False,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            num_tfilm_blocks,
            cond_dim,
            use_transformer,
        )
        self.mbconv = MBConv(
            channels_in=channels,
            channels_out=channels,
            cond_dim=cond_dim,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c, L)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Pass through MBConv: (bs, c, L) -> (bs, c, L)
        x = self.mbconv(x, cond=cond_embed)

        # Apply TFiLM: (bs, c, L) -> (bs, c, L)
        x = self.tfilm(x)

        return x


class _MidcoderTransformerLayer(nn.Module):
    """Transformer layer with MultiheadAttention for MidcoderTransformer1D."""

    def __init__(
        self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = AdaGroupNorm(8, d_model, cond_dim=dim_feedforward)
        self.norm2 = AdaGroupNorm(8, d_model, cond_dim=dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, L, d_model)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, L, d_model)
        """
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out), cond_embed)
        ff = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff), cond_embed)
        return x
