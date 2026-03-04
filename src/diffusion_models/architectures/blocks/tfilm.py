import torch
import torch.nn as nn

from .base import SinusoidalEmbedding


class TFiLM(nn.Module):
    def __init__(
        self, num_blocks: int, channels: int, rnn_hidden: int, rnn_layers: int = 1
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn = nn.LSTM(
            input_size=channels,
            hidden_size=self.rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.to_params = nn.Linear(2 * self.rnn_hidden, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (B, C, T)
        - cond: (B, cond_dim)
        Returns:
        - out: (B, C, T) with TFiLM applied
        """
        B, C, T_orig = x.shape

        # Calculate padding needed
        remainder = T_orig % self.num_blocks
        if remainder != 0:
            pad_amount = self.num_blocks - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="replicate")
        else:
            pad_amount = 0

        B, C, T = x.shape
        block_len = T // self.num_blocks

        # (B, C, T) -> (B, num_blocks, C, block_len)
        blocks = x.view(B, C, self.num_blocks, block_len).permute(0, 2, 1, 3)

        # Max-pool over time within each block: (B, num_blocks, C)
        pooled = blocks.max(dim=-1).values

        # RNN over blocks (sequence length = num_blocks)
        rnn_out, _ = self.rnn(pooled)  # (B, num_blocks, 2 * hidden)

        # Affine params per block/channel
        params = self.to_params(rnn_out)  # (B, num_blocks, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each: (B, num_blocks, C)

        # Apply affine to each block
        scale = scale.unsqueeze(-1)  # (B, num_blocks, C, 1)
        shift = shift.unsqueeze(-1)  # (B, num_blocks, C, 1)
        mod_blocks = scale * blocks + shift  # (B, num_blocks, C, block_len)

        # Reassemble to (B, C, T)
        out = mod_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # Remove padding if any was added
        if pad_amount > 0:
            out = out[:, :, :T_orig]

        return out


class TFiLMTransformer(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_blocks: int,
        channels: int,
        num_heads: int,
        num_layers: int,
        ffn_dim_multiplier: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.layers = nn.ModuleList(
            [
                _TFiLMTransformerLayer(
                    d_model=self.channels,
                    num_blocks=self.num_blocks,
                    num_heads=num_heads,
                    dim_feedforward=ffn_dim_multiplier * channels,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.to_params = nn.Linear(channels, 2 * channels)
        self.pos_encoding = SinusoidalEmbedding(channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T_orig = x.shape

        # Calculate padding needed
        remainder = T_orig % self.num_blocks
        if remainder != 0:
            pad_amount = self.num_blocks - remainder
            x = torch.nn.functional.pad(x, (0, pad_amount), mode="replicate")
        else:
            pad_amount = 0

        B, C, T = x.shape
        block_len = T // self.num_blocks

        # (B, C, T) -> (B, num_blocks, C, block_len)
        blocks = x.view(B, C, self.num_blocks, block_len).permute(0, 2, 1, 3)

        # Max-pool over time within each block: (B, num_blocks, C)
        pooled = blocks.max(dim=-1).values

        # Use positional encoding
        pos = torch.arange(self.num_blocks, device=pooled.device)  # (num_blocks,)
        pos_emb = self.pos_encoding(pos).unsqueeze(0)  # (1, num_blocks, C)
        pooled = pooled + pos_emb  # (B, num_blocks, C)

        # Transformer over blocks (sequence length = num_blocks)
        transformer_out = pooled
        for layer in self.layers:
            transformer_out = layer(transformer_out, cond)

        # Affine params per block/channel
        params = self.to_params(transformer_out)  # (B, num_blocks, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each: (B, num_blocks, C)

        # Apply affine to each block
        scale = scale.unsqueeze(-1)  # (B, num_blocks, C, 1)
        shift = shift.unsqueeze(-1)  # (B, num_blocks, C, 1)
        mod_blocks = scale * blocks + shift  # (B, num_blocks, C, block_len)

        # Reassemble to (B, C, T)
        out = mod_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # Remove padding if any was added
        if pad_amount > 0:
            out = out[:, :, :T_orig]

        return out


class _TFiLMTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_blocks, d_model = channels)
            cond: (B, cond_dim)
        Returns:
            x: (B, num_blocks, d_model)
        """
        # Apply self-attention mechanism
        attn_out, _ = self.self_attn(
            x, x, x, need_weights=False
        )  # (B, num_blocks, d_model)

        # Pass through MLP
        x = x + attn_out
        x = self.linear1(x)
        x = torch.nn.SiLU(x)
        ff = self.linear2(x)
        ff = torch.nn.SiLU(ff)
        x = x + ff
        return x
