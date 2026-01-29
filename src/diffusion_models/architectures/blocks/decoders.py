import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CrossChannelAttention,
    ResidualLayer,
    get_activation,
)
from diffusion_models.architectures.blocks.one_d_base import (
    DepthwiseConv1DExplicit,
    ResidualLayer1D,
    SeperableConv1D,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_out, cond_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - cond: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, cond)

        return x


class Decoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels_out, cond_dim)
                for _ in range(num_residual_layers)
            ]
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
        use_seperable_conv: bool = False,
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
            if use_seperable_conv:
                conv = SeperableConv1D(
                    channels_in,
                    channels_out,
                    filters_per_channel=4,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                )
            else:
                conv = nn.Conv1d(
                    channels_in,
                    channels_out,
                    kernel_size=conv_kernel_size,
                    padding=conv_padding,
                    stride=conv_stride,
                )
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                conv,
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


class HADecoder(nn.Module):
    """
    Hybrid Attention Decoder Module for 1D data.
    Attention mechanism is used twice: first to capture cross-channel dependencies and
    then to capture temporal dependencies.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_residual_layers: int,
        activation: str = "relu",
        filter_per_channel: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)

        # Define Layers
        self.upsample = self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            self.activation,
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.cc_attention = CrossChannelAttention(
            num_channels=self.channels,
            feature_dim=filter_per_channel,
            num_heads=4,
            num_layers=6,
        )

        self.pointwise_conv = nn.Linear(filter_per_channel, 1)

        temporal_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=3,
            dim_feedforward=4 * self.channels,
            batch_first=True,
        )
        self.temporal_attention = nn.TransformerEncoder(
            temporal_enc_layer, num_layers=6
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, 2*L)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_in, 2*L)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_in, 2*L) -> (bs, c_in, 2*L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Depthwise Conv: (bs, c_in, 2*L) -> (bs, c_in, 2*L, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, c_in, 2*L, filter_per_channel) -> (bs, c_in, 2*L, filter_per_channel)
        x = self.cc_attention(x)

        # Pointwise Conv to reduce feature dim: (bs, c_in, 2*L, filter_per_channel) -> (bs, c_in, 2*L, 1)
        x = self.pointwise_conv(x)
        x = x.squeeze(-1)  # (bs, c_in, 2*L)
        x = self.activation(x)

        # Temporal Attention: (bs, c_in, 2*L) -> (bs, c_in, 2*L)
        x = x.permute(0, 2, 1)  # (bs, 2*L, c_in)
        x = self.temporal_attention(x)

        x = x.permute(0, 2, 1)  # (bs, c_in, 2*L)

        return x


class HADecoderTFiLM(nn.Module):
    """
    Hybrid Attention Decoder Module for 1D data with TFiLM.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_residual_layers: int,
        activation: str = "relu",
        filter_per_channel: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)

        # Define Layers
        self.upsample = self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            self.activation,
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.cc_attention = CrossChannelAttention(
            num_channels=self.channels,
            feature_dim=filter_per_channel,
            num_heads=4,
            num_layers=6,
        )

        self.pointwise_conv = nn.Linear(filter_per_channel, 1)

        self.tfilm = TFiLMTransformer(num_blocks=8, channels=self.channels)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, 2*L)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_in, 2*L)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_in, 2*L) -> (bs, c_in, 2*L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Depthwise Conv: (bs, c_in, 2*L) -> (bs, c_in, 2*L, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, c_in, 2*L, filter_per_channel) -> (bs, c_in, 2*L, filter_per_channel)
        x = self.cc_attention(x)

        # Pointwise Conv to reduce feature dim: (bs, c_in, 2*L, filter_per_channel) -> (bs, c_in, 2*L, 1)
        x = self.pointwise_conv(x)
        x = x.squeeze(-1)  # (bs, c_in, 2*L)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_in, 2*L) -> (bs, c_in, 2*L)
        x = self.tfilm(x)

        return x


class HADecoderImproved(nn.Module):
    """
    Improved Hybrid Attention Decoder for 1D data. Mirrors HAEncoderImproved with
    enhanced temporal dependency capture and feature preservation.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_residual_layers: int,
        activation: str = "relu",
        filter_per_channel: int = 16,
    ):
        super().__init__()
        self.channels = channels
        self.activation = get_activation(activation)

        # Upsample layer
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer1D(channels, cond_dim=cond_dim)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=1,  # No downsampling in decoder
        )

        # Multi-scale cross-channel attention
        self.cc_attention = CrossChannelAttention(
            num_channels=self.channels,
            feature_dim=filter_per_channel,
            num_heads=4,
            num_layers=2,  # Reduced from 6
        )

        # Keep more information - don't collapse to 1
        self.feature_mixer = nn.Sequential(
            nn.Linear(filter_per_channel, filter_per_channel // 2),
            nn.GELU(),
            nn.Linear(filter_per_channel // 2, filter_per_channel // 4),
        )

        # Now temporal attention works on channels * reduced_features
        d_model = self.channels * (filter_per_channel // 4)

        # Enhanced temporal attention with dilated convolutions for global context
        # self.dilated_convs = nn.ModuleList(
        #     [
        #         nn.Conv1d(
        #             d_model,
        #             d_model,
        #             kernel_size=3,
        #             padding=dilation,
        #             dilation=dilation,
        #             groups=min(d_model, 8),
        #         )
        #         for dilation in [1, 2, 4]
        #     ]
        # )

        temporal_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_attention = nn.TransformerEncoder(
            temporal_enc_layer,
            num_layers=3,  # Increased depth for better global modeling
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_in, 2*L)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_in, 2*L)
        x = self.upsample(x)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Depthwise Conv: (bs, c_in, 2*L) -> (bs, c_in, 2*L, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-channel attention on rich features
        x = self.cc_attention(x)

        # Mix features but keep dimensionality
        # (bs, c_in, 2*L, filter_per_channel) -> (bs, c_in, 2*L, filter_per_channel//4)
        x = self.feature_mixer(x)

        # Flatten channels and features for temporal attention
        bs, c, seq_len, feat_dim = x.shape
        x = x.reshape(bs, seq_len, c * feat_dim)  # (bs, 2*L, c * feat//4)

        # Add dilated convolution context for multi-scale temporal understanding
        # x_T = x.transpose(1, 2)  # (bs, d_model, seq_len)
        # conv_outs = [conv(x_T).transpose(1, 2) for conv in self.dilated_convs]
        # x = sum(conv_outs) / len(conv_outs) + x  # Residual fusion with averaging

        # Temporal attention
        x = self.temporal_attention(x)

        # Reshape back
        x = x.reshape(bs, seq_len, c, feat_dim)
        x = x.permute(0, 2, 1, 3)  # (bs, c, 2*L, feat//4)

        # Final projection to (bs, c, 2*L)
        x = x.mean(dim=-1)

        return x
