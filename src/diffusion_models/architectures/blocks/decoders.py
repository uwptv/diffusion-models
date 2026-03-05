import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CBAM,
    AdaGroupNorm,
    CrossChannelAttention,
    HAResidualLayer,
    MBConv,
    ResidualBlock,
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
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(ResidualBlock(channels_in, channels_out, cond_dim))
        for _ in range(num_residual_layers - 1):
            self.res_blocks.append(ResidualBlock(channels_out, channels_out, cond_dim))

        upsample_method = get_upsampling(method)
        self.upsample = upsample_method(channels_in // 2)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in // 2, L)
        - skip: (bs, c_in // 2, 2 * L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        x = self.upsample(x)  # (bs, c_in // 2, 2*L)
        x = torch.cat([x, skip], dim=1)  # (bs, c_in, 2 *L)

        # Pass through residual blocks: (bs, c_in, 2 * L) -> (bs, c_out, 2 * L)
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
    ):
        super().__init__(
            channels_in,
            channels_out,
            method,
            num_residual_layers,
            cond_dim,
        )
        self.cbam = CBAM(channels_out, cbam_reduction_ratio, cbam_kernel_size)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in // 2, L)
        - skip: (bs, c_in // 2, 2 * L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        x = self.upsample(x)  # (bs, c_in // 2, 2*L)
        x = torch.cat([x, skip], dim=1)  # (bs, c_in, 2 *L)

        for block in self.res_blocks:
            x = block(x, cond_embed)
            # Apply CBAM after each residual block
            x = self.cbam(x)

        return x


class MBConvDecoder(Decoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        upsampling_method: str,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
    ):
        self.mbconv_blocks = nn.ModuleList()

        self.mbconv_blocks.append(
            MBConv(channels_in, channels_out, cond_dim, expansion_factor, kernel_size)
        )

        for _ in range(num_mbconv_layers - 1):
            self.mbconv_blocks.append(
                MBConv(
                    channels_out, channels_out, cond_dim, expansion_factor, kernel_size
                )
            )

        upsample_method = get_upsampling(upsampling_method)
        self.upsample = upsample_method(channels_in // 2)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in // 2, L)
        - skip: (bs, c_in // 2, 2 * L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        x = self.upsample(x)  # (bs, c_in // 2, 2*L)
        x = torch.cat([x, skip], dim=1)  # (bs, c_in, 2 *L)

        for block in self.mbconv_blocks:
            x = block(x, cond_embed)

        return x


class TFiLMDecoder(Decoder1D):
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
    ):
        super().__init__(
            channels_in,
            channels_out,
            method,
            num_residual_layers,
            cond_dim,
        )
        self.tfilm = TFiLM(
            num_blocks=num_tfilm_blocks,
            channels=channels_out,
            rnn_hidden=hidden_size_rnn,
            rnn_layers=num_layers_rnn,
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in // 2, L)
        - skip: (bs, c_in // 2, 2 * L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, 2*L)
        """
        # Upsample: (bs, c_in // 2, L) -> (bs, c_in // 2, 2*L)
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # (bs, c_in, 2 *L)

        # Pass through residual blocks: (bs, c_in, 2 * L) -> (bs, c_out, 2 * L)
        for block in self.res_blocks:
            x = block(x, cond_embed)
            # Apply TFiLM after each residual block
            x = self.tfilm(x, cond_embed)

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
        ffn_dim_multiplier: int,
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
        )
        # Replace TFiLM mechanism with Transformer Mechanism
        self.tfilm = TFiLMTransformer(
            channels=channels_out,
            num_blocks=num_tfilm_blocks,
            num_heads=num_transformer_heads,
            num_layers=1,
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

        # Refine the upsampled features with separable convolution: (bs * channels, features_out, 2*L) -> (bs * channels, features_out, 2*L)
        x = self.refinement(x)
        x = self.norm(x, cond_embed.repeat_interleave(c, dim=0))
        x = self.activation(x)

        # Update feature dimension after upsampling
        _, feat_out, seq_len = x.shape

        # Add temporal attention: (bs * channels, features_out, 2*L) -> (bs * channels, features_out, 2*L)
        x = self.temporal_attention(x, cond_embed.repeat_interleave(c, dim=0))
        # Reshape back
        x = x.reshape(bs, c, feat_out, seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, 2*L, features_out)

        # Cross-Channel Attention: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        x = self.cc_attention(x)

        # Pass through residual blocks: (bs, channels, 2*L, features_out) -> (bs, channels, 2*L, features_out)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        return x
