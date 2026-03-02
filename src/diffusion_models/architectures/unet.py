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
        class_idx: int | None = None,
        num_timesteps: int = 30,
        guidance_scale: float = 1.0,
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

        # If not provided a class label, use null class
        if class_idx is None:
            class_labels = torch.full(
                (num_samples,), 0, device=device, dtype=torch.long
            )
        else:
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
        num_samples: int,
        p_data_shape: List[int],
        class_idx: int | None = None,
        num_timesteps: int = 30,
        guidance_scale: float = 1.0,
        class_names: List[str] | None = None,
        save_path: str | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model and visualize the results.

        Args:
            - num_samples: Number of samples to generate
            - p_data_shape: Shape of the data to generate (channels, length)
            - class_idx: Class index for conditional generation (None uses null class)
            - num_timesteps: Number of timesteps for ODE simulation
            - guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            - class_names: Optional list of class names for title
            - save_path: Optional path to save the figure
            - device: Device to run on

        Returns:
            - Generated samples, shape (num_samples, *p_data_shape)
        """
        # Generate samples
        samples = self.sample(
            num_samples=num_samples,
            p_data_shape=p_data_shape,
            class_idx=class_idx,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            device=device,
        )

        # Move to CPU for visualization
        samples_cpu = samples.cpu().numpy()

        # Determine grid size
        num_channels = p_data_shape[0]
        num_cols = min(num_samples, 4)
        num_rows = (num_samples + num_cols - 1) // num_cols

        # Create figure
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False
        )

        # Get class name for title
        if class_names and class_idx is not None and class_idx > 0:
            class_name = class_names[
                class_idx - 1
            ]  # Adjust for 0-indexed unconditional
        elif class_idx == 0 or class_idx is None:
            class_name = "Unconditional"
        else:
            class_name = f"Class {class_idx}"

        # Plot each sample
        for idx in range(num_samples):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            sample = samples_cpu[idx]  # (channels, length)

            # Plot each channel
            for ch in range(num_channels):
                ax.plot(sample[ch], label=f"Channel {ch}", alpha=0.7)

            ax.set_title(f"Sample {idx + 1}", fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            if num_channels <= 5:  # Only show legend if not too many channels
                ax.legend(fontsize=8)

        # Hide empty subplots
        for idx in range(num_samples, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis("off")

        fig.suptitle(
            f"Generated Samples - {class_name} (guidance={guidance_scale})",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
            plt.show()
        else:
            plt.show()

        plt.close()

        return samples
