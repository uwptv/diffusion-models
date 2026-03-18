import torch
import torch.nn as nn

from .blocks.base import Conditioner, SinusoidalEmbedding


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        t_dim: int = 64,
        y_dim: int = 16,
    ):
        super().__init__()
        self.linear1 = nn.Linear(channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, channels)
        self.pos_enc = SinusoidalEmbedding(hidden_dim)
        self.conditioner = Conditioner(num_classes, t_dim, y_dim, cond_dim)
        self.dit_blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads, cond_dim) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, ...) the input tensor
        - t: (bs, 1, ...) the time tensor
        - y: (bs, 1) the class labels
        Returns:
        - u_t^theta(x|y): (bs, c, ...)
        """
        # Get conditioning embedding: (bs, cond_dim)
        cond_embed = self.conditioner(t, y)

        # Treat each timestep as a separate token.
        x = x.permute(0, 2, 1)  # (bs, seq_len, c)
        x = self.linear1(x)  # (bs, seq_len, hidden_dim)

        # Add sinusoidal positional encoding over sequence positions.
        _, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device)  # (seq_len,)
        pos_enc = self.pos_enc(pos).unsqueeze(0)  # (1, seq_len, hidden_dim)
        x = x + pos_enc  # (bs, seq_len, hidden_dim)

        # Pass through transformer layers
        for block in self.dit_blocks:
            x = block(x, cond_embed)  # (bs, seq_len, hidden_dim)

        # Permute back to (bs, c, seq_len)
        x = self.linear2(x)  # (bs, seq_len, c)
        x = x.permute(0, 2, 1)  # (bs, c, seq_len)

        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Linear(cond_dim, 6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, seq_len, hidden_dim)
        - cond_embed: (bs, cond_dim)
        Returns:
        - out: (bs, seq_len, hidden_dim)
        """
        factors = self.mlp(cond_embed)  # (bs, 6)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = factors.chunk(
            6, dim=-1
        ).unsqueeze(-1)  # Each (bs, 1, 1)
        res = x

        x = self.norm(x)
        x = gamma_1 * x + beta_1

        attn_out, _ = self.self_attn(x, x, x)  # (bs, seq_len, hidden_dim)
        attn_out = alpha_1 * attn_out
        x = res + attn_out

        res = x

        x = self.norm(x)
        x = gamma_2 * x + beta_2

        ffn_out = self.ffn(x)  # (bs, seq_len, hidden_dim)
        ffn_out = alpha_2 * ffn_out

        return res + ffn_out
