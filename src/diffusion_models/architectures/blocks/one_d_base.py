import torch
import torch.nn as nn

class ResidualLayer1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int,):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, cond_dim) -> (bs, channels)
        self.cond_adapter = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, channels)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, L)
        - cond: (bs, cond_dim)
        """
        res = x.clone() # (bs, c, L)

        # Initial conv block
        x = self.block1(x) # (bs, c, L)

        # Add conditioning embedding
        cond = self.cond_adapter(cond).unsqueeze(-1) # (bs, c, 1)
        x = x + cond

        # Second conv block
        x = self.block2(x) # (bs, c, L)

        # Add back residual
        x = x + res # (bs, c, L)

        return x

class Encoder1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_in, cond_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv1d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Pass through residual blocks: (bs, c_in, L) -> (bs, c_in, L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        # Downsample: (bs, c_in, L) -> (bs, c_out, L // 2)
        x = self.downsample(x)

        return x

class Midcoder1D(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels, cond_dim) for _ in range(num_residual_layers)
        ])

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

class Decoder1D(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, cond_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1)
        )
        self.res_blocks = nn.ModuleList([
            ResidualLayer1D(channels_out, cond_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, L)
        - cond_embed: (bs, cond_dim)
        """
        # Upsample: (bs, c_in, L) -> (bs, c_out, 2*L)
        x = self.upsample(x)
        
        # Pass through residual blocks: (bs, c_out, 2*L) -> (bs, c_out, 2*L)
        for block in self.res_blocks:
            x = block(x, cond_embed)

        return x