from abc import ABC
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

    @torch.no_grad()
    def visualize(
        self,
        p_data_shape: List[int],
        dataset_mean: torch.Tensor,
        dataset_std: torch.Tensor,
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

                # Apply denormalization for visualization
                sample = sample * dataset_std + dataset_mean

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

    @torch.no_grad()
    def plot_tsne(
        self,
        p_data_shape: List[int],
        real_data: List[torch.Tensor],
        dataset_mean: torch.Tensor,
        dataset_std: torch.Tensor,
        num_samples: int = 1000,
        num_timesteps: int = 30,
        guidance_scale: float = 2.0,
        perplexity: int = 30,
        class_names: List[str] | None = None,
        device: torch.device = None,
    ) -> Figure:
        """
        Generate samples and create a t-SNE plot comparing real vs generated data.

        Args:
            p_data_shape: Shape of the data (channels, length)
            real_data: List of real data tensors per class. Each tensor: (N, C, L)
            num_samples: Number of samples to generate per class
            num_timesteps: Number of ODE timesteps
            guidance_scale: Classifier-free guidance scale
            perplexity: t-SNE perplexity
            class_names: Optional class names for the legend
            device: Device to run on

        Returns:
            Matplotlib Figure with the t-SNE plot
        """
        if device is None:
            device = next(self.parameters()).device

        num_classes = len(real_data)

        all_features = []
        labels = []
        sources = []

        for class_idx in range(num_classes):
            # Subsample real data
            real = real_data[class_idx][:num_samples]
            real_flat = real.cpu().reshape(real.shape[0], -1).numpy()

            # Generate samples
            generated = self.sample(
                num_samples=num_samples,
                p_data_shape=p_data_shape,
                class_idx=class_idx,
                num_timesteps=num_timesteps,
                guidance_scale=guidance_scale,
                device=device,
            )  # (num_samples, channels, length)

            # Apply denormalization
            generated = generated * dataset_std.to(device) + dataset_mean.to(device)

            gen_flat = generated.cpu().reshape(generated.shape[0], -1).numpy()

            all_features.append(real_flat)
            labels.extend([class_idx] * len(real_flat))
            sources.extend(["real"] * len(real_flat))

            all_features.append(gen_flat)
            labels.extend([class_idx] * len(gen_flat))
            sources.extend(["generated"] * len(gen_flat))

        all_features = np.concatenate(all_features, axis=0)
        labels = np.array(labels)
        sources = np.array(sources)

        # Scale features to [0, 1] for better t-SNE performance
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)

        # Run t-SNE
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, max_iter=1000
        )
        embeddings = tsne.fit_transform(all_features)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Use more differentiable colors
        colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ]

        for class_idx in range(num_classes):
            cls_name = (
                class_names[class_idx]
                if class_names and class_idx < len(class_names)
                else f"Class {class_idx}"
            )

            color = colors[class_idx % len(colors)]

            # Real: filled circles
            mask_real = (labels == class_idx) & (sources == "real")
            ax.scatter(
                embeddings[mask_real, 0],
                embeddings[mask_real, 1],
                c=color,
                marker="o",
                alpha=0.6,
                s=30,
                edgecolors="black",
                linewidth=0.5,
                label=f"{cls_name} (real)",
            )

            # Generated: crosses (larger for visibility)
            mask_gen = (labels == class_idx) & (sources == "generated")
            ax.scatter(
                embeddings[mask_gen, 0],
                embeddings[mask_gen, 1],
                c=color,
                marker="x",
                alpha=0.6,
                s=60,
                linewidth=1.5,
                label=f"{cls_name} (generated)",
            )

        ax.set_title(
            f"t-SNE: Real vs Generated (guidance_scale={guidance_scale})", fontsize=12
        )
        ax.legend(loc="best", fontsize=9, markerscale=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        plt.close()
