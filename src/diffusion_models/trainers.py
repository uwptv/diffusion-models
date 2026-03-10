import tempfile
from abc import ABC, abstractmethod
from collections import deque
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


class SmoothLogger:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, new_val):
        self.history.append(new_val)
        # Returns the moving average
        return sum(self.history) / len(self.history)


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        stopper: EarlyStopping,
        trial: optuna.Trial | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.trial = trial
        self.stopper = stopper
        self._seed = seed
        self._gen = torch.Generator()
        if self._seed is not None:
            self._gen.manual_seed(self._seed)

    @abstractmethod
    def get_training_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_validation_loss(self, **kwargs) -> torch.Tensor:
        pass

    def _reset_generator(self, device: torch.device):
        self._gen = torch.Generator(device=device)
        if self._seed is not None:
            self._gen.manual_seed(self._seed)

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        lr: float,
        **kwargs,
    ):
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Initialize smooth logger for validation loss
        smooth_logger = SmoothLogger(window_size=5)

        # Reset generator for reproducibility of training dynamics and metrics
        self._reset_generator(device)

        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, _ in pbar:
            opt.zero_grad()

            # Compute training and validation loss
            train_loss = self.get_training_loss(**kwargs)

            # Backprop on training loss and step optimizer
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            opt.step()

            val_loss = self.get_validation_loss(**kwargs)
            loss_val = val_loss.item()

            # Smooth validation loss for better early stopping decisions
            smoothed_val_loss = smooth_logger.update(loss_val)

            # Check early stopping
            if self.trial:
                self.trial.report(smoothed_val_loss, step=idx)
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
        null_class: int,
        seed: int = 42,
        **kwargs,
    ):
        assert eta > 0 and eta < 1
        super().__init__(model, seed=seed, **kwargs)
        self.eta = eta
        self.path = path
        self.null_class = null_class

    def _sample_batch(self, batch_size: int, subset: str = "train"):
        z, y = self.path.sample_conditioning_variable(batch_size, subset=subset)

        mask = torch.rand(batch_size, device=z.device, generator=self._gen) < self.eta
        y[mask] = self.null_class

        t = torch.rand(
            (batch_size,) + (1,) * (z.ndim - 1), generator=self._gen, device=z.device
        )
        x = self.path.sample_conditional_path(z, t, generator=self._gen)
        u_t = self.path.conditional_vector_field(x, z, t)

        return x, t, y, u_t

    def get_training_loss(self, batch_size: int) -> torch.Tensor:
        x, t, y, u_t = self._sample_batch(batch_size, subset="train")
        u_t_theta = self.model(x, t, y)
        return torch.mean((u_t - u_t_theta) ** 2)

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        x, t, y, u_t = self._sample_batch(batch_size, subset="val")
        with torch.no_grad():
            u_t_theta = self.model(x, t, y)
        return torch.mean((u_t - u_t_theta) ** 2)


class TinyHARTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        path: GaussianConditionalProbabilityPath,
    ):
        self.model = model
        self.path = path

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_training_loss(self, batch_size: int) -> torch.Tensor:
        """Compute training loss from the training sampler."""
        x, labels = self.path.sample_conditioning_variable(batch_size, subset="train")
        labels = labels.squeeze(1).long()
        train_pred = self.model(x)
        return nn.CrossEntropyLoss()(train_pred, labels)

    def get_validation_loss(self, batch_size: int) -> torch.Tensor:
        """Compute validation loss from the validation sampler."""
        x, labels = self.path.sample_conditioning_variable(batch_size, subset="val")
        labels = labels.squeeze(1).long()
        with torch.no_grad():
            val_pred = self.model(x)
        return nn.CrossEntropyLoss()(val_pred, labels)

    def _compute_predictions(self, num_samples: int, device: torch.device):
        """Helper to compute predictions and labels from validation set."""
        x, labels = self.path.sample_conditioning_variable(num_samples, subset="test")
        x = x.to(device)
        labels = labels.squeeze(1).long()

        with torch.no_grad():
            preds = torch.argmax(self.model(x), dim=1).cpu()

        return labels.cpu().numpy(), preds.numpy()

    def _compute_confusion_matrix(self, num_samples: int, device: torch.device):
        labels_np, preds_np = self._compute_predictions(num_samples, device)
        return confusion_matrix(
            labels_np,
            preds_np,
        )

    def _compute_f1_score(self, num_samples: int, device: torch.device):
        labels_np, preds_np = self._compute_predictions(num_samples, device)
        return (
            f1_score(labels_np, preds_np, average="macro"),
            f1_score(labels_np, preds_np, average="weighted"),
        )

    def _compute_accuracy(self, num_samples: int, device: torch.device) -> float:
        labels_np, preds_np = self._compute_predictions(num_samples, device)
        return accuracy_score(labels_np, preds_np)

    def _save_confusion_matrix(
        self, cm, save_path: str, class_names: list[str] | None = None
    ):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Counts)")

        n_rows, n_cols = cm.shape
        if class_names:
            ticks = range(len(class_names))
            plt.xticks(ticks, class_names, rotation=45, ha="right")
            plt.yticks(ticks, class_names)
        else:
            plt.xticks(range(n_cols))
            plt.yticks(range(n_rows))

        # Write counts into each cell
        threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(n_rows):
            for j in range(n_cols):
                val = int(cm[i, j])
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, f"{val}", ha="center", va="center", color=color)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        name: str,
        lr: float = 1e-3,
        batch_size: int = 64,
        save_model: bool = False,
        confusion_matrix_samples: int | None = 1000,
        class_names: list[str] | None = None,
    ):
        """Train the TinyHAR classifier with MLflow logging."""
        mlflow.set_experiment("TinyHAR")

        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            mlflow.log_param("lr", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)

            self.model.to(device)
            opt = self.get_optimizer(lr)

            pbar = tqdm(range(num_epochs))
            for idx in pbar:
                # Training step
                self.model.train()
                opt.zero_grad()
                train_loss = self.get_training_loss(batch_size)
                train_loss.backward()
                opt.step()

                # Validation step
                self.model.eval()
                val_loss = self.get_validation_loss(batch_size)
                loss_val = val_loss.item()
                loss_train = train_loss.item()

                # Log metrics
                mlflow.log_metric("train_loss", loss_train, step=idx)
                mlflow.log_metric("val_loss", loss_val, step=idx)

                pbar.set_description(
                    f"Epoch {idx}, train: {loss_train:.4f}, val: {loss_val:.4f}"
                )

            self.model.eval()

            # Save model checkpoint
            if save_model:
                mlflow.pytorch.log_model(
                    self.model,
                    name=f"{name}_model",
                )

            # Compute and log confusion matrix
            if confusion_matrix_samples:
                cm = self._compute_confusion_matrix(confusion_matrix_samples, device)
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / f"{name}_cm.png"
                    self._save_confusion_matrix(
                        cm, str(tmp_path), class_names=class_names
                    )
                    mlflow.log_artifact(str(tmp_path), artifact_path="plots")

            # Log final metrics
            f1_macro, f1_weighted = self._compute_f1_score(
                confusion_matrix_samples, device
            )
            acc = self._compute_accuracy(confusion_matrix_samples, device)
            mlflow.log_metric("f1_score_macro", f1_macro)
            mlflow.log_metric("f1_score_weighted", f1_weighted)
            mlflow.log_metric("accuracy", acc)
