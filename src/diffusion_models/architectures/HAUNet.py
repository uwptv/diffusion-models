import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    DepthwiseConv1DExplicit,
    FeatureFusion,
)
from diffusion_models.architectures.blocks.decoders import (
    HADecoder,
)
from diffusion_models.architectures.blocks.encoders import (
    HAEncoder,
)
from diffusion_models.architectures.blocks.midcoders import Midcoder4D
from diffusion_models.architectures.unet import UNet


class HAUNet(UNet):
    """
    Hybrid Attention UNet for 1D data.
    """

    def __init__(
        self,
        cond_dim: int,
        num_residual_layers: int,
        num_encoder_decoder_layers: int,
        num_classes: int,
        input_channels: int = 3,
        initial_features: int = 8,
    ):
        super().__init__(cond_dim, num_classes)
        # initial convolution
        self.init_conv = DepthwiseConv1DExplicit(input_channels, initial_features)

        # create encoder and decoder layers
        encoders = []
        decoders = []
        for i in range(num_encoder_decoder_layers):
            encoders.append(
                HAEncoder(
                    channels=input_channels,
                    features_in=2 ** (i + 3),
                    features_out=2 ** (i + 4),
                    cond_dim=cond_dim,
                    num_residual_layers=num_residual_layers,
                )
            )
            decoders.append(
                HADecoder(
                    channels=input_channels,
                    features_in=2 ** (i + 4),
                    features_out=2 ** (i + 3),
                    cond_dim=cond_dim,
                    num_residual_layers=num_residual_layers,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        # use simple midcoder
        self.midcoder = Midcoder4D(
            2 ** (num_encoder_decoder_layers + 3), num_residual_layers, cond_dim
        )

        # final convolution
        self.final_conv = FeatureFusion(initial_features)
