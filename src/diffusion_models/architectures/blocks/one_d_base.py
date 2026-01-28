import torch
import torch.nn as nn


class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        # Converts (bs, cond_dim) -> (bs, channels)
        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, channels)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        """
        res = x.clone()  # (bs, c, L)

        # Initial conv block
        x = self.block1(x)  # (bs, c, L)

        # Add conditioning embedding
        cond = self.cond_adapter(cond).unsqueeze(-1)  # (bs, c, 1)
        x = x + cond

        # Second conv block
        x = self.block2(x)  # (bs, c, L)

        # Add back residual
        x = x + res  # (bs, c, L)

        return x


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
        x = self.depthwise(x)  # (bs, c_in, L)
        return x
