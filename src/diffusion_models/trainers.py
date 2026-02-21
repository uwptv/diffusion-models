import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import optuna
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from tqdm import tqdm

from .dynamics.base import ConditionalVectorField
from .dynamics.prob_paths import GaussianConditionalProbabilityPath


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        stopper: EarlyStopping,
        trial: optuna.Trial | None = None,
    ):
        super().__init__()
        self.model = model
        self.trial = trial
        self.stopper = stopper

    @abstractmethod
    def get_training_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_validation_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        lr: float,
        val_every: int = 10,
        **kwargs,
    ):
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, _ in pbar:
            opt.zero_grad()

            # Compute training and validation loss
            train_loss = self.get_training_loss(**kwargs)

            # Backprop on training loss and step optimizer
            train_loss.backward()
            opt.step()

            # Compute validation loss periodically
            if idx % val_every == 0:
                val_loss = self.get_validation_loss(**kwargs)
                loss_val = val_loss.item()

                # Check early stopping
                if self.trial:
                    self.trial.report(loss_val, step=idx)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                self.stopper(loss_val)
                if self.stopper.should_stop:
                    print(f"\nEarly stopping triggered at epoch {idx}")
                    mlflow.set_tag("termination_reason", "local_early_stopping")
                    mlflow.log_param("early_stopped_epoch", idx)
                    break

                mlflow.log_metric("val_loss", loss_val, step=idx)

            # Log losses to mlflow
            loss_train = train_loss.item()
            mlflow.log_metric("train_loss", loss_train, step=idx)

            # Update tqdm description
            pbar.set_description(
                f"Epoch {idx}, train_loss: {loss_train:.3f}, val_loss: {loss_val:.3f}"
            )

        self.model.eval()

        return mlflow.active_run().info.run_id, self.stopper.best_loss


class CFGTrainer(Trainer):
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        **kwargs,
    ):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def _sample_batch(self, batch_size: int):
        """Sample a batch of data for flow matching."""
        z, y = self.path.sample_conditioning_variable(batch_size)

        mask = torch.rand(batch_size) < self.eta
        y[mask] = 0

        t = torch.rand((batch_size,) + (1,) * (z.ndim - 1)).to(z.device)
        x = self.path.sample_conditional_path(z, t)
        u_t = self.path.conditional_vector_field(x, z, t)

        return x, t, y, u_t

    def get_training_loss(self, batch_size: int) -> torch.Tensor:
        x, t, y, u_t = self._sample_batch(batch_size)
        u_t_theta = self.model(x, t, y)
        return torch.mean((u_t - u_t_theta) ** 2)

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        x, t, y, u_t = self._sample_batch(batch_size)
        with torch.no_grad():
            u_t_theta = self.model(x, t, y)
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

    def get_loss(self, batch_size: int, val_split: float):
        # Sample data points and labels
        x, labels = self.path.sample_conditioning_variable(
            batch_size
        )  # (batch_size, c, x_dim), (batch_size, 1)
        # Provide cross-entropy loss with class indices
        labels = (labels.squeeze(1) - 1).long()

        # Split into training and validation sets
        split_idx = int(batch_size * (1 - val_split))
        x_train, x_val = x[:split_idx], x[split_idx:]
        labels_train, labels_val = labels[:split_idx], labels[split_idx:]

        # Regress and output loss
        train_pred = self.model(x_train)  # (batch_size, num_classes)
        val_pred = self.model(x_val)  # (batch_size, num_classes)

        # Provide cross-entropy loss with logits from the model and class indices from the data
        train_loss = nn.CrossEntropyLoss()(train_pred, labels_train)
        val_loss = nn.CrossEntropyLoss()(val_pred, labels_val)

        return train_loss, val_loss

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

    def _compute_f1_score(
        self, num_samples: int, device: torch.device
    ) -> dict[str, float]:
        """Compute F1 scores for the classifier.

        Returns:
            Dictionary with 'macro' and 'weighted' F1 scores.
        """
        x, labels = self.path.sample_conditioning_variable(num_samples)
        labels = self._normalize_labels(labels)
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).cpu()

        labels_np = labels.cpu().numpy()
        preds_np = preds.numpy()

        return f1_score(labels_np, preds_np, average="macro"), f1_score(
            labels_np, preds_np, average="weighted"
        )

    def _compute_accuracy(self, num_samples: int, device: torch.device) -> float:
        """Compute classification accuracy.

        Returns:
            Accuracy score between 0 and 1.
        """
        x, labels = self.path.sample_conditioning_variable(num_samples)
        labels = self._normalize_labels(labels)
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1).cpu()

        labels_np = labels.cpu().numpy()
        preds_np = preds.numpy()

        return accuracy_score(labels_np, preds_np)

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
        class_names: list[str] | None = None,
        **kwargs,
    ):
        with mlflow.start_run(run_name=name):
            run_id, val_loss = super().train(
                num_epochs=num_epochs, device=device, lr=lr, **kwargs
            )

            if save_path:
                if config is None:
                    config = self._default_checkpoint_config()
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / name
                    self._save_checkpoint(str(tmp_path), config)
                    mlflow.pytorch.log_model(
                        self.model,
                        name=name,
                        run_id=run_id,
                    )

            if confusion_matrix_samples:
                cm = self._compute_confusion_matrix(confusion_matrix_samples, device)
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / f"{name}_cm.png"
                    self._save_confusion_matrix(
                        cm, str(tmp_path), class_names=class_names
                    )
                    mlflow.log_artifact(
                        str(tmp_path), artifact_path="plots", run_id=run_id
                    )

            # Log F1 scores and accuracy
            f1_score_weighted, f1_score_macro = self._compute_f1_score(
                confusion_matrix_samples, device
            )
            accuracy_score = self._compute_accuracy(confusion_matrix_samples, device)
            mlflow.log_metric("f1_score_weighted", f1_score_weighted, run_id=run_id)
            mlflow.log_metric("f1_score_macro", f1_score_macro, run_id=run_id)
            mlflow.log_metric("accuracy", accuracy_score, run_id=run_id)

            return run_id, val_loss
