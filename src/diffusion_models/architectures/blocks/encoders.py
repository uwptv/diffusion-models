import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    AdaGroupNorm,
    CrossChannelAttention,
    DepthwiseConv1DExplicit,
    ResidualLayer,
    SeperableConv1D,
    get_activation,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_in, cond_dim) for _ in range(num_residual_layers)]
        )
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, cond)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Encoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        num_groups: int = 8,
        activation: str = "silu",
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv1d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )
        self.norm = AdaGroupNorm(
            num_groups=num_groups, num_channels=channels_out, cond_dim=cond_dim
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        return x


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
        conv_stride: int = 2,
        conv_padding: int = 1,
        use_transformer: bool = False,
        use_seperable_conv: bool = False,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        if use_seperable_conv:
            self.downsample = SeperableConv1D(
                channels_in=channels_in,
                channels_out=channels_out,
                filters_per_channel=4,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            )
        else:
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
        - cond_embed: (bs, cond_dim)
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


class HAEncoder(nn.Module):
    """
    Hybrid Attention Encoder for 1D data.
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
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=2,
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
        - x: (bs, c_in, L // 2)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Depthwise Conv: (bs, c_in, L) -> (bs, c_in, L // 2, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, c_in, L // 2, filter_per_channel) -> (bs, c_in, L // 2, filter_per_channel)
        x = self.cc_attention(x)

        # Pointwise Conv to reduce feature dim: (bs, c_in, L // 2, filter_per_channel) -> (bs, c_in, L // 2, 1)
        x = self.pointwise_conv(x)
        x = x.squeeze(-1)  # (bs, c_in, L // 2)
        x = self.activation(x)

        # Temporal Attention: (bs, c_in, L // 2) -> (bs, c_in, L // 2)
        x = x.permute(0, 2, 1)  # (bs, L // 2, c_in)
        x = self.temporal_attention(x)

        x = x.permute(0, 2, 1)  # (bs, c_in, L // 2)

        return x


class HAEncoderTFiLM(nn.Module):
    """
    Hybrid Attention Encoder for 1D data.
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
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=2,
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
        - x: (bs, c_in, L // 2)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Depthwise Conv: (bs, c_in, L) -> (bs, c_in, L // 2, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, c_in, L // 2, filter_per_channel) -> (bs, c_in, L // 2, filter_per_channel)
        x = self.cc_attention(x)

        # Pointwise Conv to reduce feature dim: (bs, c_in, L // 2, filter_per_channel) -> (bs, c_in, L // 2, 1)
        x = self.pointwise_conv(x)
        x = x.squeeze(-1)  # (bs, c_in, L // 2)

        # Apply TFiLM: (bs, c, L // 2) -> (bs, c, L // 2)
        x = self.tfilm(x)

        return x


class HAEncoderImproved(nn.Module):
    """
    Improved Hybrid Attention Encoder for 1D data. Replaced the pointwise conv with a channel-feature mixer to preserver more information.
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

        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )

        self.depthwise_conv = DepthwiseConv1DExplicit(
            channels_in=self.channels,
            filters_per_channel=filter_per_channel,
            kernel_size=3,
            padding=1,
            stride=2,
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
        temporal_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.temporal_attention = nn.TransformerEncoder(
            temporal_enc_layer, num_layers=3
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        Returns:
        - x: (bs, c_in, L // 2)
        """
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # (bs, c, L) -> (bs, c, L//2, filter_per_channel)
        x = self.depthwise_conv(x)
        x = self.activation(x)

        # Cross-channel attention on rich features
        x = self.cc_attention(x)

        # Mix features but keep dimensionality
        # (bs, c, L//2, filter_per_channel) -> (bs, c, L//2, filter_per_channel//4)
        x = self.feature_mixer(x)

        # Flatten channels and features for temporal attention
        bs, c, seq_len, feat_dim = x.shape
        x = x.reshape(bs, seq_len, c * feat_dim)  # (bs, L//2, c * feat//4)

        x = self.temporal_attention(x)

        # Reshape back
        x = x.reshape(bs, seq_len, c, feat_dim)
        x = x.permute(0, 2, 1, 3)  # (bs, c, L//2, feat//4)

        # Final projection to (bs, c, L//2)
        x = x.mean(dim=-1)  # or use another linear layer

        return x
