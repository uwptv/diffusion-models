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


def get_upsampling(method: str) -> nn.Module:
    """
    Returns an upsampling module based on a string name.

    Args:
        method: Name of the upsampling method

    Returns:
        nn.Module: The upsampling module
    """
    methods = {
        "transposed": TransposedConv,
        "interpolation": InterpolationConv,
        "pixel_shuffle": PixelShuffle,
    }

    method_lower = method.lower()
    if method_lower not in methods:
        raise ValueError(
            f"Unknown upsampling method: {method}. Available: {list(methods.keys())}"
        )

    return methods[method_lower]


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.half_dim, dtype=torch.float32)
            / self.half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (B,), (B, 1), (L,), or (B, L)
        if x.dim() == 1:
            x = x.view(-1, 1)  # (N, 1) where N=B or L
        angles = (
            x * self.freqs * 2 * math.pi
        )  # broadcast to (N, half_dim) or (B, L, half_dim)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin, cos], dim=-1)


class Conditioner(nn.Module):
    def __init__(self, num_classes: int, t_dim: int, y_dim: int, cond_dim: int) -> None:
        super().__init__()

        self.t_embedder = SinusoidalEmbedding(t_dim)
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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        cond_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.activation = get_activation(activation)

        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=3, padding=1)
        self.res_conv = (
            nn.Conv1d(channels_in, channels_out, kernel_size=1)
            if channels_in != channels_out
            else nn.Identity()
        )

        self.norm1 = AdaGroupNorm(num_channels=channels_in, cond_dim=cond_dim)
        self.norm2 = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)

        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            self.activation,
            nn.Linear(cond_dim, channels_out),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L) for 1D
        - cond: (bs, cond_dim)
        Returns:
        - output: same shape as x
        """
        res = x  # (bs, c_in, L)
        x = self.norm1(x, cond)
        x = self.activation(x)
        x = self.conv1(x)  # (bs, c_out, L)

        cond_adapted = self.cond_adapter(cond)  # (bs, c_out)
        for _ in range(x.ndim - 2):
            cond_adapted = cond_adapted.unsqueeze(-1)  # (bs, c_out, 1, ..., 1)
        x = x + cond_adapted

        x = self.norm2(x, cond)
        x = self.activation(x)
        x = self.conv2(x)

        x = x + self.res_conv(res)
        return x


class HAResidualLayer(ResidualBlock):
    def __init__(
        self,
        features: int,
        cond_dim: int,
    ):
        super().__init__(
            features,
            cond_dim,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, channels, L, features)
        - cond: (bs, cond_dim)
        Returns:
        - output: same shape as x
        """
        # Merge batch and channels
        bs, channels, L, features = x.shape
        x = x.permute(0, 1, 3, 2).reshape(
            bs * channels, features, L
        )  # (bs * channels, features, L)

        # Expand cond to match reshaped batch size
        # (bs, cond_dim) -> (bs*channels, cond_dim)
        cond_expanded = cond.repeat_interleave(channels, dim=0)

        x = super().forward(x, cond_expanded)  # (bs * channels, features, L)

        # Reshape back to original format
        x = x.reshape(bs, channels, features, L).permute(
            0, 1, 3, 2
        )  # (bs, channels, L, features)

        return x


class CrossChannelAttention(nn.Module):
    """
    Cross-channel interaction using multi-head self-attention.
    Applies attention across sensor channels to capture inter-channel dependencies.

    Args:
        num_channels: Number of sensor channels (C)
        feature_dim: Feature dimension per channel (F)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        num_heads = min(
            num_heads, feature_dim
        )  # Ensure num_heads does not exceed feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # MultiheadAttention operates on feature_dim (last dimension)
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms for post-norm transformer
        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(num_layers)]
        )
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(num_layers)]
        )

        # Feedforward networks
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, 4 * feature_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * feature_dim, feature_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L, F) - batch, channels, sequence_length, features_per_channel

        Returns:
            (B, C, L, F) - attended features
        """
        B, C, L, F = x.shape

        # Reshape to apply attention across channels at each time step
        # (B, C, L, F) -> (B*L, C, F)
        # This treats each timestep independently and applies cross-channel attention
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * L, C, F)  # (B*L, C, F)

        # Apply transformer layers
        for attn, norm1, ffn, norm2 in zip(
            self.attention_layers, self.norms1, self.ffns, self.norms2
        ):
            # Self-attention across channels (C is the sequence dimension)
            attn_out, _ = attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
            x_reshaped = norm1(x_reshaped + attn_out)

            # Feedforward with residual
            ffn_out = ffn(x_reshaped)
            x_reshaped = norm2(x_reshaped + ffn_out)

        # Reshape back: (B*L, C, F) -> (B, L, C, F) -> (B, C, L, F)
        output = x_reshaped.reshape(B, L, C, F).permute(0, 2, 1, 3)

        return output


class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization. Applies GroupNorm followed by a scale and shift
    conditioned on an external embedding.
    """

    def __init__(self, num_channels: int, cond_dim: int) -> None:
        super().__init__()

        # Use a max of 32 groups and a minimum of 8 groups, or channels//4 if that is in between
        num_groups = min(32, max(4, num_channels // 4))

        # Find largest divisor
        while num_channels % num_groups != 0:
            num_groups -= 1

        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)
        # outputs scale (γ) and shift (β)
        # Initialize to do nothing at start (γ ≈ 1, β ≈ 0)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
        - x: (B, C, ...)
        - cond: (B, cond_dim)
        Returns:
        - out: (B, C, ...)
        """

        x = self.group_norm(x)  # (B, C, ...)

        # If no conditioning is provided, return normalized x
        if cond is None:
            return x

        gamma, beta = self.linear(cond).chunk(2, dim=1)  # (B, C)

        # Make them dimension-independent (broadcast across any spatial dims)
        shape = [gamma.shape[0], gamma.shape[1]] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)

        # Apply scale and shift
        out = x * (1 + gamma) + beta  # (B, C, ...)

        return out


class SeperableConv1D(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        cond_dim: int,
        filters_per_channel: int,
        stride: int,
        padding: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels_in,
            filters_per_channel * channels_in,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=channels_in,
        )
        self.pointwise = nn.Conv1d(
            filters_per_channel * channels_in, channels_out, kernel_size=1
        )
        self.activation = nn.SiLU()
        self.norm1 = AdaGroupNorm(
            num_channels=filters_per_channel * channels_in, cond_dim=cond_dim
        )
        self.norm2 = AdaGroupNorm(num_channels=channels_out, cond_dim=cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond: (bs, cond_dim)
        """
        # Depthwise convolution
        x = self.depthwise(x)  # (bs, filters_per_channel * c_in, L)
        x = self.norm1(x, cond)  # (bs, filters_per_channel * c_in, L)
        x = self.activation(x)  # (bs, filters_per_channel * c_in, L)

        # Pointwise convolution
        x = self.pointwise(x)  # (bs, c_out, L)
        x = self.norm2(x, cond)  # (bs, c_out, L)
        x = self.activation(x)  # (bs, c_out, L)

        return x


class DepthwiseConv1DExplicit(nn.Module):
    """
    Explicit depthwise separable 1D convolution where all channels use the same set of filters.
    Uses batch dimension reshaping (TinyHAR-style) to apply shared weights across all input channels.

    Args:
        channels_in: Number of input channels (C)
        filters_per_channel: Number of filters to apply (F)
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Padding added to input

    Output shape: (B, C, F, L_out) where each channel produces F feature maps
    """

    def __init__(
        self,
        channels_in: int,
        cond_dim: int,
        filters_per_channel: int,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.filters_per_channel = filters_per_channel

        # Single conv layer with shared weights for all channels
        # Processes 1 input channel -> F output channels
        self.depthwise_conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters_per_channel,
            kernel_size=3,
            padding=1,
        )
        self.norm = AdaGroupNorm(num_channels=filters_per_channel, cond_dim=cond_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, L)
            cond_embed: Condition embedding of shape (B, cond_dim)

        Returns:
            Output tensor of shape (B, C, L_out, F)
        """
        B, C, L = x.shape

        # Reshape: (B, C, L) -> (B*C, 1, L)
        # Each channel becomes a separate sample in the batch
        x_reshaped = x.reshape(B * C, 1, L)

        # Apply shared convolution: (B*C, 1, L) -> (B*C, F, L_out)
        features = self.depthwise_conv(x_reshaped)

        # Expand the cond_embed to match the new batch size (B*C, cond_dim)
        cond_expanded = cond_embed.repeat_interleave(C, dim=0)

        features = self.norm(features, cond_expanded)
        features = self.activation(features)

        # Reshape back: (B*C, F, L_out) -> (B, C, L_out, F)
        _, F, L_out = features.shape
        output = features.reshape(B, C, F, L_out).permute(0, 1, 3, 2)

        return output


class FeatureFusion(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fusion_layer = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, C, L, feature_dim)
        Returns:
        - out: (B, C, L)
        """
        x = self.fusion_layer(x)  # (B, C, L, 1)
        out = x.squeeze(-1)  # (B, C, L)
        return out


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int, kernel_size: int):
        super().__init__()
        hidden = max(
            4, channels // reduction_ratio
        )  # Ensure hidden layer has at least 4 units
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # 1. Channel Attention
        # Compute the max and average across the time dimension
        avg_p = torch.mean(x, dim=2)  # (B, C)
        max_p, _ = torch.max(x, dim=2)  # (B, C)

        # Sum then Sigmoid (Original Paper Logic)
        cbam_c = self.channel_attention(avg_p) + self.channel_attention(max_p)
        cbam_c = torch.sigmoid(cbam_c).unsqueeze(-1)  # (B, C, 1)

        # Scale the input
        x = x * cbam_c

        # 2. Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        sa_input = torch.cat([avg_out, max_out], dim=1)  # (B, 2, L)
        sa = self.spatial_attention(sa_input)  # (B, 1, L)

        # Scale and add residual
        out = x * sa
        return out + residual


class MBConv(nn.Module):
    """
    Mobile (Inverted) Bottleneck Convolutional Block for 1D data.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        cond_dim: int,
        expansion_factor: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        hidden_dim = channels_in * expansion_factor
        padding = kernel_size // 2

        self.expand_conv = nn.Conv1d(channels_in, hidden_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_dim,
        )
        self.project_conv = nn.Conv1d(hidden_dim, channels_out, kernel_size=1)
        self.norm_expand = AdaGroupNorm(num_channels=hidden_dim, cond_dim=cond_dim)
        self.norm_depthwise = AdaGroupNorm(num_channels=hidden_dim, cond_dim=cond_dim)
        self.norm_project = AdaGroupNorm(
            num_channels=channels_out,
            cond_dim=cond_dim,
        )
        self.activation = nn.ReLU6()
        self.use_residual = (channels_in == channels_out) and (stride == 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, channels_in, L)
        - cond: (B, cond_dim)
        Returns:
        - out: (B, channels_out, L_out)
        """
        residual = x

        x = self.expand_conv(x)  # (B, hidden_dim, L)
        x = self.norm_expand(x, cond)  # (B, hidden_dim, L)
        x = self.activation(x)

        x = self.depthwise_conv(x)  # (B, hidden_dim, L_out)
        x = self.norm_depthwise(x, cond)  # (B, hidden_dim, L_out)
        x = self.activation(x)

        x = self.project_conv(x)  # (B, channels_out, L_out)
        x = self.norm_project(x, cond)  # (B, channels_out, L_out)

        if self.use_residual:
            x = (
                x + residual
            )  # (B, channels_out, L_out), L_out is L in that case and channels_out == channels_in

        return x


# -- Upsampling Modules -- #


class TransposedConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, channels, L)
        Returns:
        - out: (B, channels, L_out = 2 * L)
        """
        out = self.transposed_conv(x)  # (B, channels, L_out)
        return out


class InterpolationConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, channels, L)
        Returns:
        - out: (B, channels, L_out = 2 * L)
        """
        x_upsampled = nn.functional.interpolate(
            x, scale_factor=2, mode="linear", align_corners=False
        )  # (B, channels, L_out)
        out = self.conv(x_upsampled)  # (B, channels, L_out)
        return out


class PixelShuffle(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, channels, L)
        Returns:
        - out: (B, channels, L_out = 2 * L)
        """
        batch_size, _, seq_len = x.size()
        x = self.conv(x)  # (B, channels * 2, L)
        x = x.view(batch_size, self.channels, 2, seq_len)  # (B, channels, 2, L)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, channels, L, 2)
        out = x.view(batch_size, self.channels, seq_len * 2)

        return out
