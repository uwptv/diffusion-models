from abc import ABC
from typing import List

import torch

from diffusion_models.architectures.blocks.base import Conditioner
from diffusion_models.dynamics.base import CFGVectorFieldODE, ConditionalVectorField
from diffusion_models.dynamics.simulators import EulerSimulator


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

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        p_data_shape: List[int],
        class_label: int | None = None,
        num_timesteps: int = 100,
        guidance_scale: float = 1.0,
        null_class: int = 0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Draw samples from the diffusion model.

        Args:
            - num_samples: Number of samples to generate
            - p_data_shape: Shape of the data to generate
            - class_label: Class labels for conditional generation
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

        # If not provided a class label, use null class
        if class_label is None:
            class_labels = torch.full(
                (num_samples,), null_class, device=device, dtype=torch.long
            )
        else:
            class_labels = torch.full(
                (num_samples,), class_label, device=device, dtype=torch.long
            )

        # Create timesteps from t=0 to t=1
        ts = torch.linspace(0, 1, num_timesteps, device=device)
        ts = ts.reshape(1, -1, *([1] * (x0.ndim - 1)))  # (1, T, 1, ...)
        ts = ts.expand(num_samples, -1, *([1] * (x0.ndim - 1)))  # (B, T, 1, ...)

        # Create ODE and simulator
        ode = CFGVectorFieldODE(self, guidance_scale=guidance_scale)
        simulator = EulerSimulator(ode)

        # Simulate
        x1 = simulator.simulate(x0, ts, y=class_labels, null_class=null_class)
        print("Generated samples shape:", x1.shape)

        return x1
