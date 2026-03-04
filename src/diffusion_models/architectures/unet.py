from abc import ABC
from typing import List

import matplotlib.pyplot as plt
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
        self.num_classes = num_classes
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

        # Safely flatten t to (bs,) — squeeze is unsafe when bs=1
        t = t.view(x.shape[0])

        # Safely flatten y to (bs,)
        y = y.view(x.shape[0])

        # Create the conditioning embedding
        cond = self.conditioner(t, y)  # (bs, cond_dim)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, ...)

        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x, cond)
            skip_connections.append(skip.clone())

        # Midcoder
        x = self.midcoder(x, cond)

        # Decoder path with concatenative skip connections
        for decoder in self.decoders:
            skip_x = skip_connections.pop()
            x = decoder(x, skip_x, cond)

        # Final convolution
        x = self.final_conv(x)  # (bs, c, ...)

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

        # Class labels for model are 1-indexed, with 0 reserved for unconditional
        class_labels = torch.full(
            (num_samples,), class_idx + 1, device=device, dtype=torch.long
        )

        # Create timesteps from t=0 to t=1
        ts = torch.linspace(0, 1, num_timesteps, device=device)
        ts = ts.reshape(1, -1, *([1] * (x0.ndim - 1)))  # (1, T, 1, ...)
        ts = ts.expand(num_samples, -1, *([1] * (x0.ndim - 1)))  # (B, T, 1, ...)

        # Create ODE and simulator
        ode = CFGVectorFieldODE(self, guidance_scale=guidance_scale)
        simulator = EulerSimulator(ode)

        # Simulate
        x1 = simulator.simulate(x0, ts, y=class_labels)

        return x1

    @torch.no_grad()
    def visualize(
        self,
        p_data_shape: List[int],
        class_idx: int | None = None,
        num_timesteps: int = 30,
        guidance_scales: List[float] = [2.0, 3.0, 4.0],
        class_names: List[str] | None = None,
        save_path: str | None = None,
        device: torch.device = None,
    ) -> dict:
        """
        Sample one sample per (class, guidance scale) pair and visualize as a grid.
        Rows correspond to classes, columns to guidance scales.

        Args:
            - p_data_shape: Shape of the data to generate (channels, length)
            - class_idx: If provided, only plot this class. If None, plot all data classes.
            - num_timesteps: Number of timesteps for ODE simulation
            - guidance_scales: List of guidance scales — one column per scale
            - class_names: Optional list of class names for plot titles
            - save_path: Optional path to save the figure
            - device: Device to run on

        Returns:
            - dict mapping (class_idx, guidance_scale) -> generated sample tensor
        """
        if class_idx is None:
            class_indices = list(range(self.num_classes))  # external indexing: 0..K-1
        else:
            class_indices = [class_idx]

        num_rows = len(class_indices)
        num_cols = len(guidance_scales)
        num_channels = p_data_shape[0]

        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False
        )

        all_samples = {}

        for r, cls_idx in enumerate(class_indices):
            for c, gs in enumerate(guidance_scales):
                ax = axes[r, c]

                sample = self.sample(
                    num_samples=1,
                    p_data_shape=p_data_shape,
                    class_idx=cls_idx,
                    num_timesteps=num_timesteps,
                    guidance_scale=float(gs),
                    device=device,
                )  # (1, channels, length)
                all_samples[(cls_idx, float(gs))] = sample
                sample_np = sample[0].cpu().numpy()  # (channels, length)

                for ch in range(num_channels):
                    ax.plot(sample_np[ch], linewidth=1.0, label=f"Ch {ch}")

                cls_name = (
                    class_names[cls_idx]
                    if class_names is not None and 0 <= cls_idx < len(class_names)
                    else f"Class {cls_idx}"
                )

                ax.set_title(f"{cls_name} | guidance={gs}", fontsize=10)
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)
                if num_channels <= 5:
                    ax.legend(fontsize=8)

        fig.suptitle("Generated samples (1 per cell)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        plt.show()
        plt.close()

        return all_samples
