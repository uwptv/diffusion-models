import torch
import torch.nn as nn

from .base import get_activation
from .one_d_base import ResidualLayer1D


class TFiLM(nn.Module):
    def __init__(
        self, num_blocks: int, channels: int, rnn_hidden: int, rnn_layers: int = 1
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn = nn.LSTM(
            input_size=channels,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.to_params = nn.Linear(2 * self.rnn_hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T_orig = x.shape

        # Calculate padding needed
        remainder = T_orig % self.num_blocks
        if remainder != 0:
            pad_amount = self.num_blocks - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="replicate")
        else:
            pad_amount = 0

        B, C, T = x.shape
        block_len = T // self.num_blocks

        # (B, C, T) -> (B, num_blocks, C, block_len)
        blocks = x.view(B, C, self.num_blocks, block_len).permute(0, 2, 1, 3)

        # Max-pool over time within each block: (B, num_blocks, C)
        pooled = blocks.max(dim=-1).values

        # RNN over blocks (sequence length = num_blocks)
        rnn_out, _ = self.rnn(pooled)  # (B, num_blocks, 2 * hidden)

        # Affine params per block/channel
        params = self.to_params(rnn_out)  # (B, num_blocks, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each: (B, num_blocks, C)

        # Apply affine to each block
        scale = scale.unsqueeze(-1)  # (B, num_blocks, C, 1)
        shift = shift.unsqueeze(-1)  # (B, num_blocks, C, 1)
        mod_blocks = scale * blocks + shift  # (B, num_blocks, C, block_len)

        # Reassemble to (B, C, T)
        out = mod_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # Remove padding if any was added
        if pad_amount > 0:
            out = out[:, :, :T_orig]

        return out


class TFiLMTransformer(nn.Module):
    def __init__(
        self, num_blocks: int, channels: int, num_heads: int = 8, num_layers: int = 6
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=num_heads,
            dim_feedforward=4 * channels,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_params = nn.Linear(channels, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T_orig = x.shape

        # Calculate padding needed
        remainder = T_orig % self.num_blocks
        if remainder != 0:
            pad_amount = self.num_blocks - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="replicate")
        else:
            pad_amount = 0

        B, C, T = x.shape
        block_len = T // self.num_blocks

        # (B, C, T) -> (B, num_blocks, C, block_len)
        blocks = x.view(B, C, self.num_blocks, block_len).permute(0, 2, 1, 3)

        # Max-pool over time within each block: (B, num_blocks, C)
        pooled = blocks.max(dim=-1).values

        # Transformer over blocks (sequence length = num_blocks)
        transformer_out = self.transformer(pooled)  # (B, num_blocks, C)

        # Affine params per block/channel
        params = self.to_params(transformer_out)  # (B, num_blocks, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each: (B, num_blocks, C)

        # Apply affine to each block
        scale = scale.unsqueeze(-1)  # (B, num_blocks, C, 1)
        shift = shift.unsqueeze(-1)  # (B, num_blocks, C, 1)
        mod_blocks = scale * blocks + shift  # (B, num_blocks, C, block_len)

        # Reassemble to (B, C, T)
        out = mod_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # Remove padding if any was added
        if pad_amount > 0:
            out = out[:, :, :T_orig]

        return out


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
        conv_stride: int = 1,
        conv_padding: int = 1,
        use_transformer: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels_in, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
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
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(
                    channels_in,
                    channels_out,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                ),
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
