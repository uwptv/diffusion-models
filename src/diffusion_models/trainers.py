from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from .dynamics.base import ConditionalVectorField
from .dynamics.prob_paths import GaussianConditionalProbabilityPath
from .utils.sizes import MiB, model_size_b


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        lr: float = 1e-3,
        path: str = "training_loss.png",
        **kwargs,
    ) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        plot_dir = Path("loss_plots")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / Path(path).name

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses = []

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, _ in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(f"Epoch {idx}, loss: {loss.item():.3f}")

        # Finish
        self.model.eval()

        if losses:
            plt.clf()
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(
                True, which="both", alpha=0.3
            )  # Add grid for both major and minor ticks
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()


class CFGTrainer(Trainer):
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        null_label,
        **kwargs,
    ):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.sample_conditioning_variable(
            batch_size
        )  # z shape (batch_size, c, x_dim), y shape (batch_size, 1)

        # Step 2: Set each label to the null class with probability eta
        mask = torch.rand(batch_size) < self.eta
        y[mask] = self.null_label

        # Step 3: Sample t and x
        t = torch.rand((batch_size,) + (1,) * (z.ndim - 1)).to(
            z.device
        )  # (batch_size, 1, x_dim)
        x = self.path.sample_conditional_path(z, t)  # (batch_size, c, x_dim)

        # Step 4: Regress and output loss
        u_t_theta = self.model(x, t, y)  # (batch_size, c, x_dim)
        u_t = self.path.conditional_vector_field(x, z, t)  # (batch_size, c, x_dim)
        return torch.mean((u_t - u_t_theta) ** 2)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        name: str,
        lr: float = 1e-3,
        **kwargs,
    ) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        plot_dir = Path("plots/losses")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{name}.png"

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses = []

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, _ in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_description(f"Epoch {idx}, loss: {loss.item():.3f}")

        # Finish
        self.model.eval()

        if losses:
            plt.clf()
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(
                True, which="both", alpha=0.3
            )  # Add grid for both major and minor ticks
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
