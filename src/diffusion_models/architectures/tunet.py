from typing import List

import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    Conditioner,
    InitialConvSeperable,
    SeperableConv1D,
)
from diffusion_models.architectures.blocks.decoders import (
    TFiLMDecoder,
    TFiLMDecoderSeperable,
    TFiLMDecoderTransposed,
)
from diffusion_models.architectures.blocks.encoders import (
    TFiLMEncoder,
    TFiLMEncoderSeperable,
)
from diffusion_models.architectures.blocks.midcoders import TransFiLMMidcoder
from diffusion_models.architectures.tfilm_unet import TFiLMUNet


class TUNet(TFiLMUNet):
    """
    TUNet architecture for 1D signals with Transformer-based midcoder.
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_t_blocks: int,
        num_classes: int,
        cond_dim: int,
        num_transformer_layers: int = 6,
    ):
        super().__init__(
            channels, num_residual_layers, num_t_blocks, num_classes, cond_dim
        )
        self.midcoder = TransFiLMMidcoder(
            channels[-1],
            num_residual_layers,
            num_transformer_layers=num_transformer_layers,
            cond_dim=cond_dim,
        )


class TUNetTransposed(TUNet):
    """
    TUNet architecture for 1D signals with transposed convolutions in decoders for upsampling."""

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        num_t_blocks: int,
        num_classes: int,
        cond_dim: int,
        num_transformer_layers: int = 6,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            num_t_blocks,
            num_classes,
            cond_dim,
            num_transformer_layers,
        )

        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            decoders.append(
                TFiLMDecoderTransposed(
                    next_c,
                    curr_c,
                    num_residual_layers,
                    num_t_blocks,
                    cond_dim,
                )
            )
        self.decoders = nn.ModuleList(reversed(decoders))


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


class PaperTUNet(nn.Module):
    """
    TUnet architecture as described in the paper => exact implemenation doesn't work well
    """

    def __init__(self, num_classes: int, cond_dim: int, input_channels=3):
        super().__init__()
        self.conditioner = Conditioner(
            num_classes=num_classes,
            t_dim=64,
            y_dim=16,
            cond_dim=cond_dim,
        )
        self.input_channels = input_channels

        encoders = [
            TFiLMEncoder(
                channels_in=self.input_channels,
                channels_out=64,
                num_residual_layers=2,
                num_tfilm_blocks=64,
                cond_dim=cond_dim,
                conv_kernel_size=66,
                conv_stride=4,
                conv_padding=31,
            ),
            TFiLMEncoder(
                channels_in=64,
                channels_out=128,
                num_residual_layers=2,
                num_tfilm_blocks=64,
                cond_dim=cond_dim,
                conv_kernel_size=18,
                conv_stride=4,
                conv_padding=7,
            ),
        ]
        decoders = [
            TFiLMDecoder(
                channels_in=256,
                channels_out=128,
                num_residual_layers=2,
                num_tfilm_blocks=64,
                cond_dim=cond_dim,
                use_transpose_conv=True,
                conv_kernel_size=8,
                conv_stride=4,
                conv_padding=2,
                conv_output_padding=1,
                activation="leaky_relu",
            ),
            TFiLMDecoder(
                channels_in=128,
                channels_out=64,
                num_residual_layers=2,
                num_tfilm_blocks=64,
                cond_dim=cond_dim,
                use_transpose_conv=True,
                conv_kernel_size=18,
                conv_stride=4,
                conv_padding=7,
                conv_output_padding=2,
                activation="leaky_relu",
            ),
        ]
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.encoder_final_conv = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=8,
            stride=4,
            padding=2,
        )

        self.midcoder = TransFiLMMidcoder(
            channels=256,
            num_residual_layers=2,
            num_transformer_layers=6,
            cond_dim=cond_dim,
        )

        self.decoder_final_conv = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=self.input_channels,
            kernel_size=66,
            stride=4,
            padding=31,
        )
        self.final_activation = nn.Tanh()

    def forward(self, x, t, y):
        """
        Args:
        - x: (bs, 1, L)
        - t: (bs, 1, 1) -> will be squeezed to (bs,)
        - y: (bs, 1) amplitude class labels
        Returns:
        - u_t^theta(x|y): (bs, 1, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Save initial input
        x_init = x.clone()  # (bs, 3, L)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, cond)  # (bs, c, L // 4)
            residuals.append(x.clone())

        x = self.encoder_final_conv(x)  # (bs, c, L // 4)
        residuals.append(x.clone())

        x = self.midcoder(x, cond)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, cond)

        x = self.decoder_final_conv(x)
        x = self.final_activation(x)

        # Add the initial input residual connection
        x = x + x_init

        return x


class PaperTUNetAdapted(nn.Module):
    """
    TUnet architecture as described in the paper adapted to toy dataset
    """

    def __init__(self, num_classes: int, cond_dim: int, input_channels=3):
        super().__init__()
        self.conditioner = Conditioner(
            num_classes=num_classes,
            t_dim=64,
            y_dim=16,
            cond_dim=cond_dim,
        )
        self.input_channels = input_channels

        encoders = [
            TFiLMEncoder(
                channels_in=self.input_channels,
                channels_out=64,
                num_residual_layers=2,
                num_tfilm_blocks=8,
                cond_dim=cond_dim,
                conv_kernel_size=5,
                conv_stride=3,
                conv_padding=2,
            ),
            TFiLMEncoder(
                channels_in=64,
                channels_out=128,
                num_residual_layers=2,
                num_tfilm_blocks=8,
                cond_dim=cond_dim,
                conv_kernel_size=3,
                conv_stride=3,
                conv_padding=2,
            ),
        ]
        decoders = [
            TFiLMDecoder(
                channels_in=256,
                channels_out=128,
                num_residual_layers=2,
                num_tfilm_blocks=8,
                cond_dim=cond_dim,
                use_transpose_conv=True,
                conv_kernel_size=2,
                conv_stride=2,
                conv_padding=0,
                activation="leaky_relu",
            ),
            TFiLMDecoder(
                channels_in=128,
                channels_out=64,
                num_residual_layers=2,
                num_tfilm_blocks=8,
                cond_dim=cond_dim,
                use_transpose_conv=True,
                conv_kernel_size=3,
                conv_stride=3,
                conv_padding=2,
                activation="leaky_relu",
            ),
        ]
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.encoder_final_conv = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.midcoder = TransFiLMMidcoder(
            channels=256,
            num_residual_layers=2,
            num_transformer_layers=6,
            cond_dim=cond_dim,
        )

        self.decoder_final_conv = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=self.input_channels,
            kernel_size=5,
            stride=3,
            padding=2,
            output_padding=2,
        )
        self.final_activation = nn.Tanh()

    def forward(self, x, t, y):
        """
        Args:
        - x: (bs, 1, L)
        - t: (bs, 1, 1) -> will be squeezed to (bs,)
        - y: (bs, 1) amplitude class labels
        Returns:
        - u_t^theta(x|y): (bs, 1, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Save initial input
        x_init = x.clone()  # (bs, 3, L)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, cond)  # (bs, c, L // 4)
            residuals.append(x.clone())

        x = self.encoder_final_conv(x)  # (bs, c, L // 4)
        residuals.append(x.clone())

        x = self.midcoder(x, cond)

        for decoder in self.decoders:
            res = residuals.pop()
            x = x + res
            x = decoder(x, cond)

        x = self.decoder_final_conv(x)
        x = self.final_activation(x)

        # Add the initial input residual connection
        x = x + x_init

        return x
