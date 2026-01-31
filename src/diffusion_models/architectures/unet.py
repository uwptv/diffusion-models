from abc import ABC

import torch

from diffusion_models.architectures.blocks.base import Conditioner
from diffusion_models.dynamics.base import ConditionalVectorField


class UNet(ConditionalVectorField, ABC):
    def __init__(
        self,
        cond_dim: int,
        num_classes: int,
        t_dim: int = 64,
        y_dim: int = 16,
    ):
        super().__init__()
        # Create the conditioner
        self.conditioner = Conditioner(
            num_classes, t_dim=t_dim, y_dim=y_dim, cond_dim=cond_dim
        )

    def _assert_initialized(self) -> None:
        required = ["init_conv", "encoders", "midcoder", "decoders", "final_conv"]
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(
                f"UNet is missing the following required attributes: {missing}"
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
        # Ensure all required components are initialized
        self._assert_initialized()

        # Squeeze t to (bs,)
        while t.ndim > 1:
            t = t.squeeze(-1)

        # Squeeze y to (bs,)
        y = y.squeeze(-1)

        # Create the conditioning embedding
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Initial convolution
        x = self.init_conv(x, cond)  # (bs, c_0, ...)

        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x, cond)
            skip_connections.append(x.clone())

        # Midcoder
        x = self.midcoder(x, cond)

        # Decoder path with ADDITIVE skip connections
        for decoder in self.decoders:
            skip_x = skip_connections.pop()
            x = x + skip_x
            x = decoder(x, cond)

        # Final convolution
        x = self.final_conv(x)  # (bs, c, ...)

        return x
