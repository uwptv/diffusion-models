import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    AdaGroupNorm,
    CrossChannelAttention,
    MBConv,
    ResidualLayer,
    ResidualLayer4D,
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
        self.norm = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)
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
        x = self.norm(
            x, cond_embed
        )  # Commented out normalization for better comparison with other models
        x = self.activation(x)

        return x


class TFiLMEncoder(nn.Module):
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
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, cond_dim=cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.activation = get_activation(activation)
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
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        # Apply activation: (bs, c_out, L // 2) -> (bs, c_out, L // 2)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_out, L // 2) -> (bs, c_out, L // 2)
        x = self.tfilm(x, cond_embed)

        return x


class TransFiLMEncoder(TFiLMEncoder):
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


class TFiLMEncoderSeperable(TFiLMEncoder):
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
            conv_kernel_size,
            conv_stride,
            conv_padding,
            use_transformer,
        )
        # Replace downsample with SeperableConv1D
        self.downsample = SeperableConv1D(
            channels_in=channels_in,
            channels_out=channels_out,
            filters_per_channel=4,
            kernel_size=conv_kernel_size,
            stride=2,
            padding=conv_padding,
        )


class TFiLMMBConvEncoder(TFiLMEncoder):
    """
    TFiLM Encoder using MBConv for downsampling.
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
            conv_kernel_size,
            conv_stride,
            conv_padding,
            use_transformer,
        )
        # Replace downsample with MBConv
        self.downsample = MBConv(
            channels_in=channels_in,
            channels_out=channels_out,
            cond_dim=cond_dim,
            expansion_factor=4,
            kernel_size=conv_kernel_size,
            stride=2,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, L // 2)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Downsample using MBConv: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x, cond=cond_embed)

        # No activation here, as MBConv intentionally uses linear output
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
        features_in: int,
        features_out: int,
        cond_dim: int,
        num_residual_layers: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)
        self.norm = AdaGroupNorm(
            num_groups=channels, num_channels=channels, cond_dim=cond_dim
        )

        # Define Layers
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer4D(
                    features_in,
                    cond_dim,
                )
                for _ in range(num_residual_layers)
            ]
        )

        self.conv = nn.Conv1d(
            features_in, features_out, kernel_size=3, padding=1, stride=2
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
        - x: (bs, channels, L // 2, features_out)
        """
        bs, c, seq_len, feat_dim = x.shape

        # Pass through residual blocks: (bs, channels, L, features_in) -> (bs, channels, L, features_in)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Merge batch and channels for convolution
        x = x.permute(0, 1, 3, 2).reshape(
            bs * c, feat_dim, seq_len
        )  # (bs * channels, features_in, L)

        # Conv: (bs * channels, features_in, L) -> (bs * channels, features_out, L // 2)
        x = self.conv(x)

        # Reshape back to 4D
        x = x.reshape(bs, c, self.features_out, seq_len // 2).permute(
            0, 1, 3, 2
        )  # (bs, channels, L // 2, features_out)

        # Apply normalization and activation
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Cross-Channel Attention: (bs, channels, L // 2, features_out) -> (bs, channels, L // 2, features_out)
        x = self.cc_attention(x)

        # Get the updated dimensions
        bs, c, seq_len, feat_dim = x.shape

        x = x.permute(0, 1, 3, 2).reshape(
            bs * c, feat_dim, seq_len
        )  # (bs*c, features_out, L//2)

        # Temporal Attention: (bs, channels, L // 2, features_out) -> (bs, channels, L // 2, features_out)
        x = self.temporal_attention(x)

        x = x.reshape(bs, c, feat_dim, seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, L // 2, features_out)

        return x
