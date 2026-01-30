import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import AdaGroupNorm, get_activation


class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_groups: int = 8,
        activation: str = "silu",
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.norm1 = AdaGroupNorm(
            num_groups=num_groups, num_channels=channels, cond_dim=cond_dim
        )
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(
            num_groups=num_groups, num_channels=channels, cond_dim=cond_dim
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            self.activation,
            nn.Linear(cond_dim, channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        Returns:
        - output: (bs, c, L)"""
        res = x
        x = self.activation(x)
        x = self.norm1(x, cond)
        x = self.conv1(x)

        cond = self.cond_adapter(cond).unsqueeze(-1)  # (bs, c, 1)
        x = x + cond

        x = self.activation(x)
        x = self.norm2(x, cond)
        x = self.conv2(x)

        return x + res


class SeperableConv1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        filters_per_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels_in,
            filters_per_channel * channels_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=channels_in,
        )
        self.pointwise = nn.Conv1d(
            filters_per_channel * channels_in, channels_out, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        """
        x = self.depthwise(x)  # (bs, filters_per_channel * c_in, L)
        x = self.pointwise(x)  # (bs, c_out, L)
        return x


class DepthwiseConv1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        filters_per_channel: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels_in,
            filters_per_channel * channels_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=channels_in,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        """
        x = self.depthwise(x)  # (bs, filters_per_channel * c_in, L)
        return x


class DepthwiseConv1DExplicit(nn.Module):
    """
    Depthwise convolution where each filter per channel is kept as a separate
    feature dimension in the output.

    Standard depthwise conv: (bs, c_in, L) -> (bs, filters_per_channel * c_in, L)
    This version: (bs, c_in, L) -> (bs, c_in, L, filters_per_channel)

    This makes the filter dimension explicit and separable for downstream processing.
    """

    def __init__(
        self,
        channels_in: int,
        filters_per_channel: int,
        kernel_size: int = 3,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.filters_per_channel = filters_per_channel

        # Still use grouped convolution, but we'll reshape the output
        self.depthwise = nn.Conv1d(
            channels_in,
            filters_per_channel * channels_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=channels_in,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)

        Returns:
        - output: (bs, c_in, L_out, filters_per_channel)
        """
        bs, c_in, L = x.shape

        # Apply depthwise convolution
        x = self.depthwise(x)  # (bs, filters_per_channel * c_in, L_out)
        L_out = x.shape[-1]

        # Reshape to separate channel and filter dimensions
        # From: (bs, filters_per_channel * c_in, L_out)
        # To: (bs, c_in, filters_per_channel, L_out)
        x = x.view(bs, c_in, self.filters_per_channel, L_out)

        # Move filter dimension to the end
        # From: (bs, c_in, filters_per_channel, L_out)
        # To: (bs, c_in, L_out, filters_per_channel)
        x = x.permute(0, 1, 3, 2)

        return x
