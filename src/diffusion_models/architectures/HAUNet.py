import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import Conditioner
from diffusion_models.architectures.blocks.decoders import (
    HADecoder,
    HADecoderImproved,
    HADecoderTFiLM,
)
from diffusion_models.architectures.blocks.encoders import (
    HAEncoder,
    HAEncoderImproved,
    HAEncoderTFiLM,
)
from diffusion_models.architectures.blocks.midcoders import Midcoder1D


class HAUNet(nn.Module):
    """
    Hybrid Attention UNet for 1D data.
    """

    def __init__(
        self,
        cond_dim: int,
        num_residual_layers: int,
        num_encoder_decoder_layers: int,
        num_classes: int = 3,
        input_channels: int = 3,
    ):
        super().__init__()
        # initial convolution
        self.initial_conv = nn.Conv1d(input_channels, 8, kernel_size=3, padding=1)

        self.conditioner = Conditioner(
            num_classes=num_classes, t_dim=64, y_dim=16, cond_dim=cond_dim
        )

        encoders = []
        decoders = []
        for i in range(num_encoder_decoder_layers):
            encoders.append(
                HAEncoder(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
            decoders.append(
                HADecoder(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder1D(input_channels, num_residual_layers, cond_dim)

        # final convolution
        self.final_conv = nn.Conv1d(
            input_channels, input_channels, kernel_size=3, padding=1
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # initial convolution
        # x = self.initial_conv(x)  # (bs, 8, L)

        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, cond)
            skip_connections.append(x)

        # Midcoder
        x = self.midcoder(x, cond)

        # Decoder path
        for decoder in self.decoders:
            skip_x = skip_connections.pop()
            x = x + skip_x
            x = decoder(x, cond)

        # final convolution
        # x = self.final_conv(x)  # (bs, c_in, L)

        return x


class HAUNetTFiLM(nn.Module):
    """
    Hybrid Attention UNet for 1D data.
    """

    def __init__(
        self,
        cond_dim: int,
        num_residual_layers: int,
        num_encoder_decoder_layers: int,
        num_classes: int = 3,
        input_channels: int = 3,
    ):
        super().__init__()
        self.conditioner = Conditioner(
            num_classes=num_classes, t_dim=64, y_dim=16, cond_dim=cond_dim
        )

        encoders = []
        decoders = []
        for i in range(num_encoder_decoder_layers):
            encoders.append(
                HAEncoderTFiLM(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
            decoders.append(
                HADecoderTFiLM(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder1D(input_channels, num_residual_layers, cond_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, cond)
            skip_connections.append(x)

        # Midcoder
        x = self.midcoder(x, cond)

        # Decoder path
        for decoder in self.decoders:
            skip_x = skip_connections.pop()
            x = x + skip_x
            x = decoder(x, cond)

        return x


class HAUNetImproved(nn.Module):
    """
    Improved Hybrid Attention UNet for 1D data.
    """

    def __init__(
        self,
        cond_dim: int,
        num_residual_layers: int,
        num_encoder_decoder_layers: int,
        num_classes: int = 3,
        input_channels: int = 3,
    ):
        super().__init__()
        self.conditioner = Conditioner(
            num_classes=num_classes, t_dim=64, y_dim=16, cond_dim=cond_dim
        )

        encoders = []
        decoders = []
        for i in range(num_encoder_decoder_layers):
            encoders.append(
                HAEncoderImproved(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
            decoders.append(
                HADecoderImproved(
                    channels=input_channels,
                    num_residual_layers=num_residual_layers,
                    cond_dim=cond_dim,
                    filter_per_channel=2 ** (6 - i),
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder1D(input_channels, num_residual_layers, cond_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, L)
        """
        # Get unified conditioning vector
        t = t.squeeze(-1).squeeze(-1)  # (bs,)
        y = y.squeeze(-1)  # (bs,)
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Encoder path
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, cond)
            skip_connections.append(x)

        # Midcoder
        x = self.midcoder(x, cond)

        # Decoder path
        for decoder in self.decoders:
            skip_x = skip_connections.pop()
            x = x + skip_x
            x = decoder(x, cond)

        return x
