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
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_groups: int = 8,
        activation: str = "silu",
        use_1d: bool = True,
    ):
        super().__init__()
        self.activation = get_activation(activation)

        # Choose convolution type
        conv_cls = nn.Conv1d if use_1d else nn.Conv2d

        self.norm1 = AdaGroupNorm(
            num_groups=num_groups, num_channels=channels, cond_dim=cond_dim
        )
        self.conv1 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(
            num_groups=num_groups, num_channels=channels, cond_dim=cond_dim
        )
        self.conv2 = conv_cls(channels, channels, kernel_size=3, padding=1)

        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            self.activation,
            nn.Linear(cond_dim, channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L) for 1D or (bs, c, H, W) for 2D
        - cond: (bs, cond_dim)
        Returns:
        - output: same shape as x
        """
        res = x
        x = self.conv1(x)
        x = self.norm1(x, cond)
        x = self.activation(x)

        cond_adapted = self.cond_adapter(cond)  # (bs, c)
        for _ in range(x.ndim - 2):
            cond_adapted = cond_adapted.unsqueeze(-1)  # (bs, c, 1, ..., 1)
        x = x + cond_adapted

        x = self.conv2(x)
        x = self.norm2(x, cond)
        x = self.activation(x)

        return x + res


class CrossChannelAttention(nn.Module):
    """
    Self-attention mechanism across channels using feature vectors.

    Input: (batch_size, c_in, sequence_length, features)
    Output: (batch_size, c_in, sequence_length, features)

    For each timestep and batch, applies self-attention across the channel dimension,
    where each channel is represented by its feature vector.
    """

    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim

        # TransformerEncoderLayer expects: (seq_len, batch, embedding_dim)
        # We'll treat channels as sequence length
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (batch_size, c_in, sequence_length, features)

        Returns:
        - output: (batch_size, c_in, sequence_length, features)
        """
        batch_size, c_in, seq_len, feature_dim = x.shape

        # Reshape to (batch_size * sequence_length, c_in, features)
        # This way we apply attention independently for each timestep
        x = x.permute(0, 2, 1, 3)  # (batch_size, sequence_length, c_in, features)
        x = x.reshape(
            batch_size * seq_len, c_in, feature_dim
        )  # (batch_size * sequence_length, c_in, features)

        # Apply self-attention across channels
        x = self.transformer(x)  # (batch_size * sequence_length, c_in, features)

        # Reshape back to original format
        x = x.reshape(batch_size, seq_len, c_in, feature_dim)
        x = x.permute(0, 2, 1, 3)  # (batch_size, c_in, sequence_length, features)

        return x


class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization. Applies GroupNorm followed by a scale and shift
    conditioned on an external embedding.
    """

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int) -> None:
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)
        # outputs scale (γ) and shift (β)
        # Initialize to do nothing at start (γ ≈ 1, β ≈ 0)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, C, ...)
        - cond: (B, cond_dim)
        Returns:
        - out: (B, C, ...)
        """

        x = self.group_norm(x)  # (B, C, ...)

        gamma, beta = self.linear(cond).chunk(2, dim=1)  # (B, C)

        # Make them dimension-independent (broadcast across any spatial dims)
        shape = [gamma.shape[0], gamma.shape[1]] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        # Apply scale and shift
        out = x * (1 + gamma) + beta  # (B, C, ...)

        return out


class InitialConvolution(nn.Module):
    """
    Provides an initial convolutional layer that extends the channels to a specified output dimension. Uses 1D or 2D convolution based on the input flag.
    Dataflow: Convolution -> Adaptive Group Normalization -> Activation
    Dimensions: Input (B, in_channels, L) -> Output (B, out_channels, L) when use_1d is True or (B, out_channels, H, W) when use_1d is False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        use_1d: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        if use_1d:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ada_group_norm = AdaGroupNorm(
            num_groups=8, num_channels=out_channels, cond_dim=cond_dim
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, in_channels, L) when use_1d is True or (B, in_channels, H, W) when use_1d is False
        - cond: (B, cond_dim)
        Returns:
        - out: (B, out_channels, L) when use_1d is True or (B, out_channels, H, W) when use_1d is False
        """
        x = self.conv(x)  # (B, out_channels, ...)
        x = self.ada_group_norm(x, cond)  # (B, out_channels, ...)
        x = self.activation(x)  # (B, out_channels, ...)
        return x


class InitialConvSeperable(InitialConvolution):
    """
    Initial convolutional layer using separable convolutions to extend channels to a specified output dimension.
    Dataflow: Separable Convolution -> Adaptive Group Normalization -> Activation
    Dimensions: Input (B, in_channels, L) -> Output (B, out_channels, L)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        filters_per_channel: int = 4,
        activation: str = "silu",
    ):
        super().__init__(
            in_channels, out_channels, cond_dim, use_1d=True, activation=activation
        )
        self.conv = SeperableConv1D(
            channels_in=in_channels,
            channels_out=out_channels,
            filters_per_channel=filters_per_channel,
            kernel_size=3,
            padding=1,
        )


class SeperableConv1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        filters_per_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
