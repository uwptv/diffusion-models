import math

import torch
import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """
    Returns an activation function module based on a string name.

    Args:
        activation: Name of the activation function

    Returns:
        nn.Module: The activation function
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),  # SiLU and Swish are the same
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
    }

    activation_lower = activation.lower()
    if activation_lower not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. Available: {list(activations.keys())}"
        )

    return activations[activation_lower]


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (B,) or (B, 1)
        Returns:
        - emb: (B, dim)
        """
        t = t.view(-1, 1)  # (B, 1)

        # Compute frequencies: [1, 10000^(2i/d)]
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.half_dim, dtype=torch.float32)
            / self.half_dim
        ).to(t.device)  # (half_dim,)

        angles = t * freqs * 2 * math.pi  # (B, half_dim)

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)

        return emb


class Conditioner(nn.Module):
    def __init__(self, num_classes: int, t_dim: int, y_dim: int, cond_dim: int) -> None:
        super().__init__()

        self.t_embedder = SinusoidalTimeEmbedding(t_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim)

        self.mlp = nn.Sequential(
            nn.Linear(t_dim + y_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: list of timesteps (B,)
        - y: list of class labels (B,)
        Returns:
        - cond: conditioning consisting of a combination of time and class embeddings (B, cond_dim)
        """

        t_embed = self.t_embedder(t)  # (B, t_dim)
        y_embed = self.y_embedder(y)  # (B, y_dim)
        cond = torch.cat([t_embed, y_embed], dim=1)  # (B, t_dim + y_dim)
        cond = self.mlp(cond)  # (B, cond_dim)

        return cond


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # Converts (bs, cond_dim) -> (bs, channels)
        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, channels)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, ...)
        - cond: (bs, cond_dim)
        """
        res = x.clone()  # (bs, c, ...)

        # Initial conv block
        x = self.block1(x)  # (bs, c, ...)

        # Add conditioning embedding - dimension independent
        cond_adapted = self.cond_adapter(cond)  # (bs, channels)

        # Add singleton dimensions to match x's dimensions after channel dim
        for _ in range(x.ndim - 2):  # x.ndim - 2 gives spatial dimensions
            cond_adapted = cond_adapted.unsqueeze(-1)
        x = x + cond_adapted

        # Second conv block
        x = self.block2(x)  # (bs, c, ...)

        # Add back residual
        x = x + res  # (bs, c, ...)

        return x


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


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels, cond_dim) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, cond)

        return x


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
