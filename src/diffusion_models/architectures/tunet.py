from typing import List

import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    InitialConvSeperable,
    SeperableConv1D,
)
from diffusion_models.architectures.blocks.decoders import (
    TFiLMDecoderSeperable,
)
from diffusion_models.architectures.blocks.encoders import (
    TFiLMEncoderSeperable,
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
        num_transformer_layers: int,
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
            num_transformer_layers=num_transformer_layers,
            ffn_expansion_factor=ffn_expansion_factor,
        )


class TUNetSeperable(TUNet):
    """
    TUNet architecture with separable convolutions for 1D signals
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_t_blocks: int,
        num_classes: int,
        cond_dim: int,
        input_channels=3,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            num_t_blocks,
            num_classes,
            cond_dim,
        )
        self.init_conv = InitialConvSeperable(
            input_channels,
            channels[0],
            cond_dim,
            filters_per_channel=4,
        )

        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TFiLMEncoderSeperable(
                    channels_in=curr_c,
                    channels_out=next_c,
                    num_residual_layers=num_residual_layers,
                    num_tfilm_blocks=num_t_blocks,
                    cond_dim=cond_dim,
                )
            )
            decoders.append(
                TFiLMDecoderSeperable(
                    channels_in=next_c,
                    channels_out=curr_c,
                    num_residual_layers=num_residual_layers,
                    num_tfilm_blocks=num_t_blocks,
                    cond_dim=cond_dim,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.final_conv = SeperableConv1D(
            channels_in=channels[0],
            channels_out=input_channels,
            filters_per_channel=4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
