from typing import List

import torch.nn as nn

from .blocks.base import InitialConvolution
from .blocks.decoders import TFiLMDecoder, TFiLMMBConvDecoder
from .blocks.encoders import TFiLMEncoder, TFiLMMBConvEncoder
from .blocks.midcoders import CBAMMidcoder, TFiLMMBConvMidcoder, TFiLMMidcoder
from .unet import UNet


class TFiLMUNet(UNet):
    """
    UNet with TFiLM conditioning for 1D signals
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_tfilm_blocks: int,
        num_classes: int,
        cond_dim: int,
        input_channels: int = 3,
    ):
        super().__init__(cond_dim, num_classes)
        self.init_conv = InitialConvolution(
            input_channels, channels[0], cond_dim, use_1d=True
        )

        # Encoders and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TFiLMEncoder(
                    curr_c, next_c, num_residual_layers, num_tfilm_blocks, cond_dim
                )
            )
            decoders.append(
                TFiLMDecoder(
                    next_c, curr_c, num_residual_layers, num_tfilm_blocks, cond_dim
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMidcoder(
            channels[-1], num_residual_layers, num_tfilm_blocks, cond_dim
        )
        self.final_conv = nn.Conv1d(
            channels[0], input_channels, kernel_size=3, padding=1
        )


class TFiLMUNetTransformer(UNet):
    """
    UNet with TFiLM conditioning for 1D signals (using Transformer-based TFiLM blocks)
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_tfilm_blocks: int,
        num_classes: int,
        cond_dim: int,
        input_channels: int = 3,
    ):
        super().__init__(cond_dim, num_classes)
        self.init_conv = InitialConvolution(
            input_channels, channels[0], cond_dim, use_1d=True
        )

        # Encoders and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TFiLMEncoder(
                    curr_c,
                    next_c,
                    num_residual_layers,
                    num_tfilm_blocks,
                    cond_dim,
                    use_transformer=True,
                )
            )
            decoders.append(
                TFiLMDecoder(
                    next_c,
                    curr_c,
                    num_residual_layers,
                    num_tfilm_blocks,
                    cond_dim,
                    use_transformer=True,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMidcoder(
            channels[-1],
            num_residual_layers,
            num_tfilm_blocks,
            cond_dim,
            use_transformer=True,
        )
        self.final_conv = nn.Conv1d(
            channels[0], input_channels, kernel_size=3, padding=1
        )


class TFiLMUNetCBAM(TFiLMUNet):
    """
    UNet with TFiLM and CBAM conditioning midcoder for 1D signals
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_tfilm_blocks: int,
        num_classes: int,
        cond_dim: int,
        input_channels: int = 3,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            num_tfilm_blocks,
            num_classes,
            cond_dim,
            input_channels,
        )

        self.midcoder = CBAMMidcoder(channels[-1], num_residual_layers, cond_dim)


class TFiLMMBConvUNet(TFiLMUNet):
    """
    UNet with TFiLM conditioning and MBConv blocks in the encoder, decoder and midcoder for 1D signals
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_tfilm_blocks: int,
        num_classes: int,
        cond_dim: int,
        input_channels: int = 3,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            num_tfilm_blocks,
            num_classes,
            cond_dim,
            input_channels,
        )

        # Encoders and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                TFiLMMBConvEncoder(
                    curr_c, next_c, num_residual_layers, num_tfilm_blocks, cond_dim
                )
            )
            decoders.append(
                TFiLMMBConvDecoder(
                    next_c, curr_c, num_residual_layers, num_tfilm_blocks, cond_dim
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = TFiLMMBConvMidcoder(
            channels[-1], num_residual_layers, num_tfilm_blocks, cond_dim
        )
