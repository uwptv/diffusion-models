from typing import List

import torch
import torch.nn as nn

from diffusion_models.dynamics.base import CFGVectorFieldODE
from diffusion_models.dynamics.simulators import EulerSimulator

from .blocks.base import Conditioner, FullHeadSelfAttention, SinusoidalEmbedding


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
        # Safely flatten t to (bs,) — squeeze is unsafe when bs=1
        t = t.view(x.shape[0])

        # Safely flatten y to (bs,)
        y = y.view(x.shape[0])

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

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        p_data_shape: List[int],
        class_idx: int,
        num_timesteps: int = 30,
        guidance_scale: List[float] = [2.0, 3.0, 4.0],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Draw samples from the diffusion model.

        Args:
            - num_samples: Number of samples to generate
            - p_data_shape: Shape of the data to generate
            - class_idx: Class index for conditional generation
            - num_timesteps: Number of timesteps for ODE simulation
            - guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            - null_class: null class
            - device: Device to run on

        Returns:
            - Generated samples, shape (num_samples, *p_data_shape)
        """

        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # Initialize from noise
        x0 = torch.randn(num_samples, *p_data_shape, device=device)

        class_labels = torch.full(
            (num_samples,), class_idx, device=device, dtype=torch.long
        )

        # Create timesteps from t=0 to t=1
        ts = torch.linspace(0, 1, num_timesteps, device=device)
        ts = ts.reshape(1, -1, *([1] * (x0.ndim - 1)))  # (1, T, 1, ...)
        ts = ts.expand(num_samples, -1, *([1] * (x0.ndim - 1)))  # (B, T, 1, ...)

        # Create ODE and simulator
        ode = CFGVectorFieldODE(
            self, guidance_scale=guidance_scale, null_class=self.num_classes
        )
        simulator = EulerSimulator(ode)

        # Simulate
        x1 = simulator.simulate(x0, ts, y=class_labels)

        return x1


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.self_attn = FullHeadSelfAttention(hidden_dim, num_heads)
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
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = [
            f.unsqueeze(1) for f in factors.chunk(6, dim=-1)
        ]  # Each (bs, 1, 1)
        res = x

        x = self.norm(x)
        x = gamma_1 * x + beta_1

        attn_out = self.self_attn(x)  # (bs, seq_len, hidden_dim)
        attn_out = alpha_1 * attn_out
        x = res + attn_out

        res = x

        x = self.norm(x)
        x = gamma_2 * x + beta_2

        ffn_out = self.ffn(x)  # (bs, seq_len, hidden_dim)
        ffn_out = alpha_2 * ffn_out

        return res + ffn_out
