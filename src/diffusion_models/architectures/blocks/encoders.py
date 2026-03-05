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
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Encoder1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()

        # First residual block expands channels: channels_in -> channels_out
        self.res_blocks.append(ResidualBlock(channels_in, channels_out, cond_dim))

        # Remaining blocks keep channels fixed: channels_out -> channels_out
        for _ in range(num_residual_layers - 1):
            self.res_blocks.append(ResidualBlock(channels_out, channels_out, cond_dim))

        self.downsample = nn.Conv1d(
            channels_out, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_out, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Save skip connection
        skip = x.clone()  # (bs, c_out, L)

        # Downsample: (bs, c_out, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        return x, skip


class CBAMEncoder(Encoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
    ):
        super().__init__(channels_in, channels_out, num_residual_layers, cond_dim)
        self.cbam = CBAM(channels_out, cbam_reduction_ratio, cbam_kernel_size)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, L // 2)
        - skip: (bs, c_out, L)
        """
        for block in self.res_blocks:
            x = block(x, cond_embed)  # ( bs, c_out, L)
            # Enhance features with CBAM
            x = self.cbam(x)

        skip = x.clone()

        x = self.downsample(x)

        return x, skip


class MBConvEncoder(Encoder1D):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
    ):
        self.mbconv_blocks = nn.ModuleList()

        # First block is a MBConvBlock that doubles channels
        self.mbconv_blocks.append(
            MBConv(
                channels_in,
                channels_out,
                cond_dim,
                expansion_factor,
                kernel_size,
            )
        )

        # Remaining blocks are MBConvBlocks that keep channels fixed
        for _ in range(num_mbconv_layers - 1):
            self.mbconv_blocks.append(
                MBConv(
                    channels_out,
                    channels_out,
                    cond_dim,
                    expansion_factor,
                    kernel_size,
                )
            )

        self.downsample = nn.Conv1d(
            channels_out, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        for block in self.mbconv_blocks:
            x = block(x, cond_embed)  # ( bs, c_out, L)

        skip = x.clone()

        x = self.downsample(x)

        return x, skip


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
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        # First residual block expands channels: channels_in -> channels_out
        self.res_blocks.append(ResidualBlock(channels_in, channels_out, cond_dim))

        # Remaining blocks keep channels fixed: channels_out -> channels_out
        for _ in range(num_residual_layers - 1):
            self.res_blocks.append(ResidualBlock(channels_out, channels_out, cond_dim))
        self.tfilm = TFiLM(
            num_blocks=num_tfilm_blocks,
            channels=channels_out,
            rnn_hidden=hidden_size_rnn,
            rnn_layers=num_layers_rnn,
        )
        self.downsample = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        Returns:
        - x: (bs, c_out, L // 2)
        - skip: (bs, c_out, L)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_out, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)
            # Enhance features with TFiLM after each residual block
            x = self.tfilm(x, cond_embed)

        skip = x.clone()  # (bs, c_out, L)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        return x, skip


class TransFiLMEncoder(TFiLMEncoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        num_transformer_heads: int,
        ffn_dim_multiplier: int,
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            64,  # Use dummy values for RNN params since they won't be used in this variant
            1,
        )
        # Replace TFiLM mechanism with Transformer Mechanism
        self.tfilm = TFiLMTransformer(
            num_blocks=num_tfilm_blocks,
            channels=channels_out,
            num_heads=num_transformer_heads,
            num_layers=1,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )


class SeperableTFiLMEncoder(TFiLMEncoder):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        cond_dim: int,
        num_residual_layers: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        filters_per_channel: int,
        activation: str = "silu",
    ):
        super().__init__(
            channels_in,
            channels_out,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn,
            num_layers_rnn,
            activation,
        )
        # Replace downsample with SeperableConv1D
        self.downsample = SeperableConv1D(
            channels_in,
            channels_out,
            cond_dim,
            filters_per_channel,
            stride=2,
        )
        self.activation = (
            nn.Identity()
        )  # Remove activation here since SeperableConv1D already has it

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Downsample using SeperableConv1D: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x, cond_embed)

        # Apply TFiLM: (bs, c_out, L // 2) -> (bs, c_out, L // 2)
        x = self.tfilm(x, cond_embed)

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
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
        num_cc_heads: int,
        num_cc_layers: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.cond_dim = cond_dim
        self.activation = get_activation(activation)
        self.norm = AdaGroupNorm(num_channels=features_out, cond_dim=cond_dim)

        # Define Layers
        self.res_blocks = nn.ModuleList(
            [
                HAResidualLayer(
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
        - x: (bs, channels, L // 2, features_out)
        """
        bs, channels, seq_len, feat_in = x.shape

        # Pass through residual blocks: (bs, channels, L, features_in) -> (bs, channels, L, features_in)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)

        # Merge batch and channels for convolution
        x = x.reshape(bs * channels, seq_len, feat_in).permute(
            0, 2, 1
        )  # (bs * channels, features_in, L)

        # Conv: (bs * channels, features_in, L) -> (bs * channels, features_out, L // 2)
        x = self.conv(x)
        x = self.norm(
            x, cond_embed.repeat_interleave(channels, dim=0)
        )  # (bs * channels, features_out, L // 2)
        x = self.activation(x)

        # Apply temporal attention first to avoid too much reshaping
        x = self.temporal_attention(
            x, cond_embed
        )  # (bs * channels, features_out, L // 2)

        _, feat_out, seq_len = x.shape

        # Reshape back to 4D
        x = x.reshape(bs, channels, feat_out, seq_len).permute(
            0, 1, 3, 2
        )  # (bs, channels, L // 2, features_out)

        # Cross-Channel Attention: (bs, channels, L // 2, features_out) -> (bs, channels, L // 2, features_out)
        x = self.cc_attention(x)

        return x
