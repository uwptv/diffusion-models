import torch
import torch.nn as nn

from diffusion_models.architectures.blocks.base import (
    CBAM,
    HAResidualLayer,
    MBConv,
    ResidualBlock,
    SinusoidalEmbedding,
)
from diffusion_models.architectures.blocks.tfilm import TFiLM, TFiLMTransformer


class Midcoder1D(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, channels, cond_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x


class CBAMMidcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        cbam_reduction_ratio: int,
        cbam_kernel_size: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, channels, cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.cbam = CBAM(channels, cbam_reduction_ratio, cbam_kernel_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        """
        # Pass through residual blocks and CBAM blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond)
            x = self.cbam(x)

        return x


class MBConvMidcoder(Midcoder1D):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        num_mbconv_layers: int,
        expansion_factor: int,
        kernel_size: int,
    ):
        self.res_blocks = nn.ModuleList()

        # Add MBConv layers after the residual layers
        for _ in range(num_mbconv_layers):
            self.res_blocks.append(
                MBConv(
                    channels,
                    channels,
                    cond_dim,
                    expansion_factor,
                    kernel_size,
                )
            )


class HAMidcoder(Midcoder1D):
    def __init__(
        self,
        features: int,
        num_residual_layers: int,
        cond_dim: int,
    ):
        super().__init__(features, num_residual_layers, cond_dim)
        self.res_blocks = nn.ModuleList(
            [HAResidualLayer(features, cond_dim) for _ in range(num_residual_layers)]
        )


class TFiLMMidcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        hidden_size_rnn: int,
        num_layers_rnn: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, channels, cond_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.tfilm = TFiLM(
            num_blocks=num_tfilm_blocks,
            channels=channels,
            rnn_hidden=hidden_size_rnn,
            rnn_layers=num_layers_rnn,
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c, L) -> (bs, c, L)
        for block in self.res_blocks:
            x = block(x, cond=cond_embed)
            x = self.tfilm(x, cond=cond_embed)

        return x


class TransFiLMMidcoder(TFiLMMidcoder):
    """
    Midcoder that uses self-attention via a transformer across the time domain for 1D signals. Used to capture long-range time dependencies.
    """

    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        cond_dim: int,
        num_tfilm_blocks: int,
        number_transformer_heads: int,
        ffn_dim_multiplier: int,
    ):
        super().__init__(
            channels,
            num_residual_layers,
            cond_dim,
            num_tfilm_blocks,
            hidden_size_rnn=channels
            * 2,  # Use dummy value since we won't use the RNN in this midcoder
            num_layers_rnn=1,
        )
        self.tfilm = TFiLMTransformer(
            cond_dim,
            num_tfilm_blocks,
            channels,
            number_transformer_heads,
            1,
            ffn_dim_multiplier,
        )


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        bs, c, seq_len = x.shape
        h = self.heads

        # 1. Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)  #
        q, k, v = map(lambda t: t.view(bs, h, -1, seq_len), qkv)

        # 2. The Performer Trick: Apply positive kernel (ReLU is a common choice)
        # Instead of Softmax(QK^T), we do (Kernel(Q) @ (Kernel(K)^T @ V))
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # 3. Linear Attention Math: O(L) instead of O(L^2)
        # Calculate the context matrix (K^T @ V) first
        # k: (b, h, d, l) -> v: (b, h, d, l)
        context = torch.einsum("bhdl,bhel->bhde", k, v)

        # Calculate the denominator for normalization
        k_sum = k.sum(dim=-1)

        # Apply context to Queries
        out = torch.einsum("bhdl,bhde->bhel", q, context)
        out = out / (torch.einsum("bhdl,bhd->bhl", q, k_sum).unsqueeze(-2) + 1e-9)

        # 4. Collapse heads and project out
        out = out.reshape(bs, -1, seq_len)
        return self.to_out(out)


class TransformerMidcoder(Midcoder1D):
    """
    Midcoder that uses self attention across the time domain for 1D signals. Used to capture long-range time dependencies. Input vectors are channel vectors at each time step, so attention is applied across the time dimension.
    """

    def __init__(
        self,
        channels,
        num_residual_layers: int,
        cond_dim: int,
        num_heads: int,
        num_transformer_layers: int,
        ffn_expansion_factor: int,
    ):
        super().__init__(channels, num_residual_layers, cond_dim)
        self.pos_enc = SinusoidalEmbedding(channels)

        # Create lists for each component
        self.transformer_layers = nn.ModuleList(
            [
                _TransformerLayer(channels, num_heads, ffn_expansion_factor)
                for _ in range(num_transformer_layers)
            ]
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond_embed: (bs, cond_dim)
        """
        x = super().forward(x, cond_embed)  # Pass through residual layers first

        _, _, seq_len = x.shape

        # Apply positional encoding once
        pos = torch.arange(seq_len, device=x.device)  # (L,)
        pos_enc = self.pos_enc(pos)  # (L, c)
        pos_enc = pos_enc.transpose(0, 1).unsqueeze(0)  # (1, c, L)
        x = x + pos_enc  # (bs, c, L)

        # Permute for multihead attention which expects (bs, seq_len, channels)
        x = x.permute(0, 2, 1)  # (bs, L, c)

        # Apply transformer layers sequentially
        for layer in self.transformer_layers:
            x = layer(x)

        # Permute back to (bs, c, L)
        x = x.permute(0, 2, 1)

        return x


class _TransformerLayer(nn.Module):
    def __init__(self, channels, num_heads, ffn_expansion_factor):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * ffn_expansion_factor),
            nn.GELU(),
            nn.Linear(channels * ffn_expansion_factor, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # Self-attention with residual
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(attn_output + x)

        # MLP with residual
        mlp_output = self.mlp(x)
        x = self.norm2(mlp_output + x)
        return x
