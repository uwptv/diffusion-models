from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
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


class TinyHARTrainer(Trainer):
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: ConditionalVectorField,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int):
        # Sample data points and labels
        x, labels = self.path.sample_conditioning_variable(
            batch_size
        )  # (batch_size, c, x_dim), (batch_size, 1)
        # Provide cross-entropy loss with class indices
        labels = (labels.squeeze(1) - 1).long()

        # Regress and output loss
        pred = self.model(x)  # (batch_size, num_classes)

        # Provide cross-entropy loss with logits from the model and class indices from the data
        return nn.CrossEntropyLoss()(pred, labels)

    def _default_checkpoint_config(self) -> dict:
        config = {}
        for key in ("input_channels", "window_size", "num_classes", "num_filters"):
            if hasattr(self.model, key):
                config[key] = getattr(self.model, key)
        return config

    def _save_checkpoint(self, save_path: str, config: dict) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "config": config}, save_path
        )

    def _get_num_classes(self) -> int | None:
        if hasattr(self.model, "num_classes"):
            return self.model.num_classes
        if hasattr(self.model, "classifier") and hasattr(
            self.model.classifier, "out_features"
        ):
            return self.model.classifier.out_features
        return None

    def _normalize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.squeeze(1).long()
        num_classes = self._get_num_classes()
        if num_classes is not None:
            min_val = int(labels.min().item())
            max_val = int(labels.max().item())
            if min_val == 1 and max_val <= num_classes:
                labels = labels - 1
        return labels

    def _compute_confusion_matrix(self, num_samples: int, device: torch.device):
        x, labels = self.path.sample_conditioning_variable(num_samples)
        labels = self._normalize_labels(labels)
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).cpu()

        num_classes = self._get_num_classes()
        labels_np = labels.cpu().numpy()
        preds_np = preds.numpy()
        return confusion_matrix(
            labels_np,
            preds_np,
            labels=list(range(num_classes)) if num_classes is not None else None,
        )

    def _save_confusion_matrix(
        self, cm, save_path: str, class_names: list[str] | None = None
    ):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if class_names:
            ticks = range(len(class_names))
            plt.xticks(ticks, class_names, rotation=45, ha="right")
            plt.yticks(ticks, class_names)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        name: str,
        lr: float = 1e-3,
        save_path: str | None = None,
        config: dict | None = None,
        confusion_matrix_samples: int | None = 1000,
        confusion_matrix_path: str | None = None,
        class_names: list[str] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        super().train(num_epochs=num_epochs, device=device, name=name, lr=lr, **kwargs)

        if save_path:
            if config is None:
                config = self._default_checkpoint_config()
            self._save_checkpoint(save_path, config)

        if confusion_matrix_samples:
            cm = self._compute_confusion_matrix(confusion_matrix_samples, device)
            if confusion_matrix_path is None:
                confusion_matrix_path = str(
                    Path("plots/confusion_matrices") / f"{name}.png"
                )
            self._save_confusion_matrix(
                cm, confusion_matrix_path, class_names=class_names
            )
