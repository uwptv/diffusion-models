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
from diffusion_models.architectures.blocks.midcoders import HAMidcoder
from diffusion_models.architectures.unet import UNet


class HAUNet(UNet):
    """
    Hybrid Attention UNet for 1D data.
    """

    def __init__(
        self,
        input_channels: int,
        initial_features: int,
        levels: int,
        upsampling_method: str,
        cond_dim: int,
        num_classes: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        num_cc_heads: int,
        num_cc_layers: int,
    ):
        super().__init__(cond_dim, num_classes)
        # initial convolution
        self.init_conv = DepthwiseConv1DExplicit(
            input_channels, cond_dim, initial_features
        )  # (bs, input_channels, L, initial_features)

        features = [initial_features]
        for _ in range(levels):
            # Double features at each level
            features.append(features[-1] * 2)

        # create encoder and decoder layers
        encoders = []
        decoders = []
        for curr_f, next_f in zip(features[:-1], features[1:]):
            encoders.append(
                HAEncoder(
                    input_channels,
                    curr_f,
                    next_f,
                    cond_dim,
                    num_residual_layers,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                    num_cc_heads,
                    num_cc_layers,
                )
            )
            decoders.append(
                HADecoder(
                    input_channels,
                    next_f,
                    curr_f,
                    upsampling_method,
                    cond_dim,
                    num_residual_layers,
                    num_tfilm_blocks,
                    hidden_size_rnn,
                    num_layers_rnn,
                    num_cc_heads,
                    num_cc_layers,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        # use custom midcoder
        self.midcoder = HAMidcoder(features[-1], num_residual_layers, cond_dim)

        # fuse features from feature dimension to output shape
        self.final_conv = FeatureFusion(initial_features)
