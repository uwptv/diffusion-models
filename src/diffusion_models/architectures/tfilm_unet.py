import torch.nn as nn

from .blocks.base import AdaGroupNorm
from .blocks.decoders import TFiLMDecoder, TransFiLMDecoder
from .blocks.encoders import TFiLMEncoder, TransFiLMEncoder
from .blocks.midcoders import (
    TFiLMMidcoder,
    TransFiLMMidcoder,
)
from .unet import UNet


class TFiLMUNet(UNet):
    """
    UNet with TFiLM conditioning for 1D signals
    """

    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        levels: int,
        upsampling_method: str,
        num_residual_layers: int,
        num_classes: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
    ):
        super().__init__(cond_dim, num_classes)
        self.init_conv = nn.Conv1d(
            input_channels, initial_channels, kernel_size=3, padding=1
        )

        # Double channels every level
        channels = [initial_channels]
        for _ in range(levels):
            channels.append(channels[-1] * 2)

        # Encoders and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TFiLMEncoder(
                    curr_c,
                    next_c,
                    num_residual_layers,
                    cond_dim,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                )
            )
            decoders.append(
                TFiLMDecoder(
                    2 * next_c,
                    curr_c,
                    upsampling_method,
                    num_residual_layers,
                    cond_dim,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMidcoder(
            channels[-1],
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn,
            num_layers_rnn,
        )
        self.final_conv = nn.Sequential(
            AdaGroupNorm(num_channels=initial_channels, cond_dim=cond_dim),
            nn.SiLU(),
            nn.Conv1d(initial_channels, input_channels, kernel_size=3, padding=1),
        )


class TFiLMUNetTransformer(UNet):
    """
    UNet with TFiLM conditioning for 1D signals (using Transformer-based TFiLM blocks)
    """

    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        levels: int,
        upsampling_method: str,
        num_residual_layers: int,
        num_classes: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        num_transformer_heads: int,
        ffn_dim_multiplier: int,
    ):
        super().__init__(cond_dim, num_classes)
        self.init_conv = nn.Conv1d(
            input_channels, initial_channels, kernel_size=3, padding=1
        )

        # Double channels every level
        channels = [initial_channels]
        for _ in range(levels):
            channels.append(channels[-1] * 2)

        # Encoders and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TransFiLMEncoder(
                    curr_c,
                    next_c,
                    num_residual_layers,
                    cond_dim,
                    num_tfilm_blocks,
                    num_transformer_heads,
                    ffn_dim_multiplier,
                )
            )
            decoders.append(
                TransFiLMDecoder(
                    next_c,
                    curr_c,
                    upsampling_method,
                    num_residual_layers,
                    cond_dim,
                    num_tfilm_blocks,
                    num_transformer_heads,
                    ffn_dim_multiplier,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TransFiLMMidcoder(
            channels[-1],
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            num_transformer_heads,
            ffn_dim_multiplier,
        )
        self.final_conv = nn.Sequential(
            AdaGroupNorm(num_channels=channels[0], cond_dim=cond_dim),
            nn.SiLU(),
            nn.Conv1d(channels[0], input_channels, kernel_size=3, padding=1),
        )
