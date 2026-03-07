import torch.nn as nn

from diffusion_models.architectures.blocks.decoders import (
    SeperableTFiLMDecoder,
)
from diffusion_models.architectures.blocks.encoders import (
    SeperableTFiLMEncoder,
)
from diffusion_models.architectures.blocks.midcoders import TransformerMidcoder
from diffusion_models.architectures.tfilm_unet import TFiLMUNet


class TUNet(TFiLMUNet):
    """
    TUNet architecture for 1D signals with Transformer-based midcoder.
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
        num_heads: int,
        ffn_expansion_factor: int,
    ):
        super().__init__(
            input_channels,
            initial_channels,
            levels,
            upsampling_method,
            num_residual_layers,
            num_classes,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn,
            num_layers_rnn,
        )
        self.midcoder = TransformerMidcoder(
            channels=initial_channels * (2**levels),
            num_residual_layers=num_residual_layers,
            cond_dim=cond_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
        )


class SeperableTUNet(TUNet):
    """
    TUNet architecture with separable convolutions for 1D signals
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
        num_heads: int,
        ffn_expansion_factor: int,
        filters_per_channel: int,
    ):
        super().__init__(
            input_channels,
            initial_channels,
            levels,
            upsampling_method,
            num_residual_layers,
            num_classes,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn,
            num_layers_rnn,
            num_heads,
            ffn_expansion_factor,
        )
        # Double channels every level
        channels = [initial_channels]
        for _ in range(levels):
            channels.append(channels[-1] * 2)

        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                SeperableTFiLMEncoder(
                    curr_c,
                    next_c,
                    cond_dim,
                    num_residual_layers,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                    filters_per_channel,
                )
            )
            decoders.append(
                SeperableTFiLMDecoder(
                    2 * next_c,
                    curr_c,
                    upsampling_method,
                    num_residual_layers,
                    cond_dim,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                    filters_per_channel,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
