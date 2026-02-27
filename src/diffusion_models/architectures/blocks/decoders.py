import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CBAM,
    AdaGroupNorm,
    CrossChannelAttention,
    HAResidualLayer,
    MBConv,
    ResidualLayer,
    SeperableConv1D,
    get_activation,
    get_upsampling,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Decoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        method: str,
        num_residual_layers: int,
        cond_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        upsample_method = get_upsampling(method)
        self.upsample = upsample_method(channels_in, channels_out)
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, cond_dim, use_1d=True)
                for _ in range(num_residual_layers)
            ]
        )
        self.norm = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)
        self.activation = get_activation(activation)
        self.refinement = nn.Conv1d(
            channels_out, channels_out, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Refine the upsampled features
        x = self.refinement(x)
        x = self.norm(x, cond_embed)
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
        method: str,
        num_residual_layers: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in, channels_out, method, num_residual_layers, cond_dim, activation
        )
        self.cbam = CBAM(channels_out, cbam_reduction_ratio, cbam_kernel_size)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Refine the upsampled features
        x = self.cbam(x, cond_embed)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x


class MBConvDecoder(Decoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        upsampling_method: str,
        num_residual_layers: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in,
            channels_out,
            upsampling_method,
            num_residual_layers,
            cond_dim,
            activation,
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
            for _ in range(num_mbconv_layers)
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample, Normalize, Activate
        x = self.upsample(x)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Pass through MBConv layers
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
        method: str,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        activation: str = "silu",
    ):
        super().__init__()
        upsampling_method = get_upsampling(method)
        self.upsample = upsampling_method(channels_in, channels_out)
        self.norm = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)
        self.activation = get_activation(activation)

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
        self.refinement = nn.Conv1d(
            channels_out, channels_out, kernel_size=3, padding=1
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
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Refine the upsampled features
        x = self.refinement(x)
        x = self.norm(x, cond_embed)
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
        method: str,
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
            method,
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


class SeperableTFiLMDecoder(TFiLMDecoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        method: str,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        filters_per_channel: int,
    ):
        super().__init__(
            channels_in,
            channels_out,
            method,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn,
            num_layers_rnn,
        )
        self.refinement = SeperableConv1D(
            channels_out,
            channels_out,
            cond_dim,
            filters_per_channel,
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
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Refine the upsampled features with separable convolution
        x = self.refinement(x, cond_embed)
        x = self.norm(x, cond_embed)
        x = self.activation(x)

        # Apply TFiLM: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        x = self.tfilm(x, cond_embed)

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
        method: str,
        cond_dim: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        num_cc_heads: int,
        num_cc_layers: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)
        self.norm = AdaGroupNorm(num_channels=features_out, cond_dim=cond_dim)

        # Define Layers
        upsample_method = get_upsampling(method)
        self.upsample = upsample_method(features_in, features_out)
        self.refinement = nn.Conv1d(
            features_out, features_out, kernel_size=3, padding=1
        )

        self.res_blocks = nn.ModuleList(
            [
                HAResidualLayer(
                    features_out,
                    cond_dim,
                )
                for _ in range(num_residual_layers)
            ]
        )

        self.cc_attention = CrossChannelAttention(
            channels,
            features_out,
            num_cc_heads,
            num_cc_layers,
        )

        self.temporal_attention = TFiLM(
            num_tfilm_blocks, features_out, hidden_size_rnn, num_layers_rnn
        )

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
        x = self.norm(x, cond_embed.repeat_interleave(c, dim=0))
        x = self.activation(x)

        # Refine the upsampled features with separable convolution
        x = self.refinement(x)
        x = self.norm(x, cond_embed.repeat_interleave(c, dim=0))
        x = self.activation(x)

        # Update feature dimension after upsampling
        feat_dim = x.shape[1]

        # Reshape back
        x = x.reshape(bs, c, feat_dim, 2 * seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, 2*L, features_out)

        # Cross-Channel Attention: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        x = self.cc_attention(x)

        x = x.permute(0, 1, 3, 2).reshape(
            bs * c, feat_dim, 2 * seq_len
        )  # (bs * channels, features_out, 2*L)
        # temporal attention: (bs * channels, features_out, 2*L) -> (bs * channels, features_out, 2*L)
        x = self.temporal_attention(x, cond_embed.repeat_interleave(c, dim=0))

        x = x.reshape(bs, c, feat_dim, 2 * seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, 2*L, features_out)

        # Pass through residual blocks: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        return x
