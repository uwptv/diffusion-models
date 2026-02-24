import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CBAM,
    AdaGroupNorm,
    CrossChannelAttention,
    DepthwiseConv1DExplicit,
    MBConv,
    ResidualLayer,
    ResidualLayer4D,
    SeperableConv1D,
    get_activation,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Decoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.norm = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        x = self.norm(
            x, cond_embed
        )  # Commented out normalization for better comparison with other models
        x = self.activation(x)

        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x


class CBAMDecoder(Decoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in, channels_out, num_residual_layers, cond_dim, activation
        )
        self.cbam = CBAM(channels_out, cbam_reduction_ratio, cbam_kernel_size)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        # Pass through base encoder block
        x = super().forward(x, cond_embed)

        # Enhance with CBAM: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.cbam(x)

        return x


class MBConvDecoder(Decoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in, channels_out, num_residual_layers, cond_dim, activation
        )
        self.upsample_layer = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.mbconv = MBConv(
            channels_in=channels_in,
            channels_out=channels_out,
            cond_dim=cond_dim,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            stride=1,
        )
        self.mbconv_stack = nn.ModuleList(
            MBConv(
                channels_in=channels_out,
                channels_out=channels_out,
                cond_dim=cond_dim,
                expansion_factor=expansion_factor,
                kernel_size=kernel_size,
                stride=1,
            )
            for _ in range(num_mbconv_layers - 1)
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample_layer(x)

        # Pass through MBConv layers
        x = self.mbconv(x, cond=cond_embed)
        for mbconv in self.mbconv_stack:
            x = mbconv(x, cond=cond_embed)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x


class TFiLMDecoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                channels_in,
                channels_out,
                kernel_size=3,
                padding=1,
            ),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.tfilm = TFiLM(
            num_blocks=num_tfilm_blocks,
            channels=channels_out,
            rnn_hidden=hidden_size_rnn,
            rnn_layers=num_layers_rnn,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)

        # Apply activation: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.tfilm(x, cond_embed)

        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x


class TransFiLMDecoder(TFiLMDecoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        num_transformer_heads: int,
        num_transformer_layers: int,
        ffn_dim_multiplier: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            64,  # Use dummy values for RNN params since they won't be used in this variant
            1,
            activation,
        )
        # Replace TFiLM mechanism with Transformer Mechanism
        self.tfilm = TFiLMTransformer(
            cond_dim=cond_dim,
            channels=channels_out,
            num_blocks=num_tfilm_blocks,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )


class TFiLMDecoderTransposed(TFiLMDecoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        conv_output_padding: int = 1,
        use_transformer: bool = False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            num_tfilm_blocks,
            cond_dim,
            use_transformer=use_transformer,
        )
        self.upsample = nn.ConvTranspose1d(
            channels_in,
            channels_out,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            output_padding=conv_output_padding,
        )


class TFiLMDecoderSeperable(TFiLMDecoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        use_transformer: bool = False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            num_tfilm_blocks,
            cond_dim,
            use_transformer=use_transformer,
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            SeperableConv1D(
                channels_in,
                channels_out,
                filters_per_channel=4,
                kernel_size=conv_kernel_size,
                padding=conv_padding,
                stride=conv_stride,
            ),
        )


class TFiLMMBConvDecoder(TFiLMDecoder):
    """
    TFiLM Decoder using MBConv for upsampling.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        cond_dim: int,
        activation: str = "silu",
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_padding: int = 1,
        use_transformer: bool = False,
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            num_tfilm_blocks,
            cond_dim,
            activation,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            use_transformer=use_transformer,
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
        )
        self.mbconv = MBConv(
            channels_in=channels_in,
            channels_out=channels_out,
            cond_dim=cond_dim,
            expansion_factor=4,
            kernel_size=conv_kernel_size,
            stride=1,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_in, 2*L)
        x = self.upsample(x)

        # MBConv: (bs, c_in, 2*L) -> (bs, c_out, 2*L)
        x = self.mbconv(x, cond=cond_embed)

        # No activation here, as MBConv includes it
        # Apply TFiLM: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.tfilm(x)

        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

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
        features_in: int,
        features_out: int,
        cond_dim: int,
        num_residual_layers: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)
        self.norm = AdaGroupNorm(
            num_groups=channels, num_channels=channels, cond_dim=cond_dim
        )

        # Define Layers
        self.upsample = self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(features_in, features_out, kernel_size=3, padding=1),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer4D(
                    features_out,
                    cond_dim,
                )
                for _ in range(num_residual_layers)
            ]
        )

        self.cc_attention = CrossChannelAttention(
            feature_dim=features_out,
            num_heads=4,
            num_layers=6,
        )

        self.temporal_attention = TFiLM(8, features_out, rnn_hidden=128)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, channels, L, features_in)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, channels, 2*L, features_out)
        """
        bs, c, seq_len, feat_dim = x.shape
        # Merge batch and channels for upsampling
        x = x.permute(0, 1, 3, 2).reshape(
            bs * c, feat_dim, seq_len
        )  # (bs * channels, features_in, L)

        # Upsample: (bs * channels, features_in, L) -> (bs * channels, features_out, 2*L)
        x = self.upsample(x)

        # Update feature dimension after upsampling
        feat_dim = x.shape[1]

        # Reshape back
        x = x.reshape(bs, c, feat_dim, 2 * seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, 2*L, features_out)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        x = self.cc_attention(x)

        x = x.permute(0, 1, 3, 2).reshape(
            bs * c, feat_dim, 2 * seq_len
        )  # (bs * channels, features_out, 2*L)
        # temporal attention: (bs * channels, features_out, 2*L) -> (bs * channels, features_out, 2*L)
        x = self.temporal_attention(x)

        x = x.reshape(bs, c, feat_dim, 2 * seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, 2*L, features_out)

        # Pass through residual blocks: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

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
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
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
                ResidualLayer(channels, cond_dim=cond_dim, use_1d=True)
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
