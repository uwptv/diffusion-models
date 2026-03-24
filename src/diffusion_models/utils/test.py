from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from tqdm import tqdm
from whar_datasets import (
    KFoldSplitter,
    Loader,
    PostProcessingPipeline,
    PreProcessingPipeline,
    WHARDatasetID,
    get_dataset_cfg,
)
from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split

from diffusion_models.architectures.tiny_har import TinyHAR

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass
class FoldResult:
    fold_id: str
    train_loss: list[float]
    val_loss: list[float]
    test_loss: float
    accuracy: float
    f1_macro: float
    y_true: list[int]
    y_pred: list[int]


@dataclass(frozen=True)
class RunConfig:
    dataset: WHARDatasetID = WHARDatasetID.UCI_HAR
    datasets_dir: str = "./notebooks/datasets"
    output_dir: str = "./outputs"
    k_folds: int = 5
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    val_percentage: float = 0.2
    parallelize: bool = True
    device: str = "cuda"  # one of: auto, cpu, cuda, mps
    show_batch_progress: bool = False


# Edit this block to control runs.
RUN_CONFIG = RunConfig()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_required_metadata(
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> None:
    required_session_cols = {"session_id", "subject_id", "activity_id"}
    missing_session_cols = required_session_cols.difference(session_df.columns)
    if missing_session_cols:
        raise RuntimeError(
            f"Missing required session metadata columns: {sorted(missing_session_cols)}"
        )

    required_activity_cols = {"activity_id", "activity_name"}
    missing_activity_cols = required_activity_cols.difference(activity_df.columns)
    if missing_activity_cols:
        raise RuntimeError(
            f"Missing required activity metadata columns: {sorted(missing_activity_cols)}"
        )

    if session_df["subject_id"].isna().any() or session_df["activity_id"].isna().any():
        raise RuntimeError(
            "Subject identifiers or activity labels contain missing values."
        )

    if activity_df["activity_name"].isna().any():
        raise RuntimeError("Activity names contain missing values.")

    if "session_id" not in window_df.columns:
        raise RuntimeError("Missing session_id in window metadata.")

    unknown_sessions = set(window_df["session_id"]).difference(
        set(session_df["session_id"])
    )
    if unknown_sessions:
        raise RuntimeError(
            "Window metadata references unknown session_ids. "
            f"Count: {len(unknown_sessions)}"
        )


def build_model(sample_shape: Sequence[int], num_classes: int) -> TinyHAR:
    if len(sample_shape) != 2:
        raise RuntimeError(
            "Expected window samples with shape (window_size, channels). "
            f"Received: {tuple(sample_shape)}"
        )

    window_size = int(sample_shape[0])
    input_channels = int(sample_shape[1])
    print(
        f"Building model with input_channels={input_channels}, window_size={window_size}, num_classes={num_classes}"
    )

    return TinyHAR(
        input_channels=input_channels,
        window_size=window_size,
        num_classes=num_classes,
    )


def to_tensor_batch(
    activity_labels: list[int],
    samples: list[list[np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_np = np.stack([sample[0] for sample in samples], axis=0).astype(np.float32)
    y_np = np.array(activity_labels, dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return y, x


def get_present_classes(
    loader: Loader, indices: list[int], class_ids: list[int]
) -> list[int]:
    present: list[int] = []
    for activity_id in class_ids:
        filtered = loader.filter_indices(indices=indices, activity_id=activity_id)
        if filtered:
            present.append(activity_id)
    if not present:
        raise RuntimeError("No classes present in provided indices.")
    return present


def sample_balanced_batch(
    loader: Loader,
    train_indices: list[int],
    present_classes: list[int],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts: dict[int, int] = {}
    num_classes = len(present_classes)

    if batch_size >= num_classes:
        base = batch_size // num_classes
        remainder = batch_size % num_classes
        for i, activity_id in enumerate(present_classes):
            counts[activity_id] = base + (1 if i < remainder else 0)
    else:
        selected = rng.choice(np.array(present_classes), size=batch_size, replace=False)
        for activity_id in selected.tolist():
            counts[int(activity_id)] = 1

    activity_labels: list[int] = []
    samples: list[list[np.ndarray]] = []

    for activity_id, n_samples in counts.items():
        if n_samples <= 0:
            continue

        sampled_y, _, sampled_x = loader.sample_items(
            batch_size=n_samples,
            indices=train_indices,
            activity_id=activity_id,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        activity_labels.extend(sampled_y)
        samples.extend(sampled_x)

    if not activity_labels:
        raise RuntimeError("Balanced sampler produced an empty batch.")

    order = np.arange(len(activity_labels))
    rng.shuffle(order)
    shuffled_labels = [activity_labels[int(i)] for i in order]
    shuffled_samples = [samples[int(i)] for i in order]

    y, x = to_tensor_batch(
        shuffled_labels, shuffled_samples, device=torch.device("cpu")
    )
    return y, x


def iter_index_batches(
    indices: list[int],
    batch_size: int,
) -> Iterable[list[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def run_train_epoch_balanced(
    model: nn.Module,
    loader: Loader,
    train_indices: list[int],
    present_classes: list[int],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    epoch_seed: int,
    show_batch_progress: bool,
    progress_desc: str,
) -> float:
    model.train(True)

    steps_per_epoch = max(1, int(np.ceil(len(train_indices) / batch_size)))
    rng = np.random.default_rng(epoch_seed)

    total_loss = 0.0
    total_examples = 0

    step_iter: Iterable[int]
    if show_batch_progress:
        step_iter = tqdm(range(steps_per_epoch), desc=progress_desc, leave=False)
    else:
        step_iter = range(steps_per_epoch)

    for _ in step_iter:
        y_cpu, x_cpu = sample_balanced_batch(
            loader=loader,
            train_indices=train_indices,
            present_classes=present_classes,
            batch_size=batch_size,
            rng=rng,
        )

        y = y_cpu.to(device)
        x = x_cpu.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        curr_batch_size = y.shape[0]
        total_loss += float(loss.item()) * curr_batch_size
        total_examples += curr_batch_size

    return total_loss / total_examples if total_examples > 0 else 0.0


def evaluate_indices(
    model: nn.Module,
    loader: Loader,
    indices: list[int],
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_indices in iter_index_batches(indices, batch_size):
            activity_labels: list[int] = []
            samples: list[list[np.ndarray]] = []
            for idx in batch_indices:
                activity_id, _, sample = loader.get_item(int(idx))
                activity_labels.append(activity_id)
                samples.append(sample)

            y_cpu, x_cpu = to_tensor_batch(
                activity_labels=activity_labels,
                samples=samples,
                device=torch.device("cpu"),
            )
            y = y_cpu.to(device)
            x = x_cpu.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            curr_batch_size = y.shape[0]
            total_loss += float(loss.item()) * curr_batch_size
            total_examples += curr_batch_size
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    return avg_loss, y_true, y_pred


def plot_loss_curves(results: list[FoldResult], output_path: Path) -> None:
    train_curves = np.array([r.train_loss for r in results], dtype=np.float64)
    val_curves = np.array([r.val_loss for r in results], dtype=np.float64)

    mean_train = train_curves.mean(axis=0)
    mean_val = val_curves.mean(axis=0)

    epochs = np.arange(1, len(mean_train) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_train, label="Train Loss", linewidth=2)
    plt.plot(epochs, mean_val, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TinyHAR K-Fold Loss Curves (Balanced Loader Sampling)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("TinyHAR Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def run_fold(
    split: Split,
    cfg: WHARConfig,
    loader: Loader,
    num_classes: int,
    class_ids: list[int],
    epochs: int,
    device: torch.device,
    show_batch_progress: bool,
) -> tuple[FoldResult, nn.Module]:
    sample_shape = loader.get_sample(int(split.train_indices[0]))[0].shape
    model = build_model(sample_shape=sample_shape, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_classes = get_present_classes(loader, split.train_indices, class_ids)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    epoch_iter = tqdm(range(epochs), desc=f"{split.identifier} epochs", leave=False)
    for epoch_idx in epoch_iter:
        train_loss = run_train_epoch_balanced(
            model=model,
            loader=loader,
            train_indices=split.train_indices,
            present_classes=train_classes,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=cfg.batch_size,
            epoch_seed=cfg.seed + epoch_idx,
            show_batch_progress=show_batch_progress,
            progress_desc=f"{split.identifier} train e{epoch_idx + 1}/{epochs}",
        )
        val_loss, _, _ = evaluate_indices(
            model=model,
            loader=loader,
            indices=split.val_indices,
            criterion=criterion,
            device=device,
            batch_size=cfg.batch_size,
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        epoch_iter.set_postfix(
            train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}"
        )

    test_loss, y_true, y_pred = evaluate_indices(
        model=model,
        loader=loader,
        indices=split.test_indices,
        criterion=criterion,
        device=device,
        batch_size=cfg.batch_size,
    )

    if y_true:
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        acc = 0.0
        f1 = 0.0

    return (
        FoldResult(
            fold_id=split.identifier,
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            test_loss=float(test_loss),
            accuracy=acc,
            f1_macro=f1,
            y_true=y_true,
            y_pred=y_pred,
        ),
        model,
    )


def main() -> None:
    run_cfg = RUN_CONFIG

    cfg = get_dataset_cfg(run_cfg.dataset, datasets_dir=run_cfg.datasets_dir)
    if cfg.dataset_id == "wisdm":
        cfg.window_time = 6.0
    print(f"Using dataset: {cfg.dataset_id} with window_time={cfg.window_time}s")
    cfg.parallelize = bool(run_cfg.parallelize)
    cfg.num_folds = int(run_cfg.k_folds)
    cfg.num_epochs = int(run_cfg.epochs)
    cfg.batch_size = int(run_cfg.batch_size)
    cfg.learning_rate = float(run_cfg.learning_rate)
    cfg.seed = int(run_cfg.seed)
    cfg.val_percentage = float(run_cfg.val_percentage)

    device = resolve_device(run_cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Using device: {device}")
    print(f"Dataset: {cfg.dataset_id}")

    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run()
    validate_required_metadata(activity_df, session_df, window_df)

    print(
        "Metadata checks passed: "
        f"subjects={session_df['subject_id'].nunique()}, "
        f"activities={session_df['activity_id'].nunique()}, "
        f"sessions={session_df['session_id'].nunique()}"
    )

    splitter = KFoldSplitter(cfg)
    splits = splitter.get_splits(session_df, window_df)

    # Build sample cache once for all folds.
    post_pipeline = PostProcessingPipeline(
        cfg=cfg,
        pre_processing_pipeline=pre_pipeline,
        window_df=window_df,
        indices=window_df.index.to_list(),
    )
    samples = post_pipeline.run()

    loader = Loader(
        session_df=session_df,
        window_df=window_df,
        samples_dir=post_pipeline.samples_dir,
        samples_dict=samples,
    )

    output_dir = Path(run_cfg.output_dir) / f"{cfg.dataset_id}_balanced_loader"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = sorted(activity_df["activity_id"].unique().tolist())
    label_to_name = dict(zip(activity_df["activity_id"], activity_df["activity_name"]))
    class_names = [str(label_to_name[label]) for label in class_ids]

    fold_results: list[FoldResult] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    per_fold_confusion_paths: dict[str, str] = {}
    best_fold_id: str | None = None
    best_f1 = float("-inf")
    best_model_for_logging: nn.Module | None = None

    mlflow.set_experiment("TinyHAR-Balanced-Loader")
    mlflow.start_run(run_name=f"tinyhar_{cfg.dataset_id}_balanced_loader")
    mlflow.log_params(
        {
            "dataset": str(cfg.dataset_id),
            "k_folds": int(cfg.num_folds),
            "epochs": int(cfg.num_epochs),
            "batch_size": int(cfg.batch_size),
            "learning_rate": float(cfg.learning_rate),
            "seed": int(cfg.seed),
            "device": str(device),
        }
    )

    split_iter = tqdm(splits, desc="K-Folds")
    for fold_idx, split in enumerate(split_iter):
        fold_result, trained_model = run_fold(
            split=split,
            cfg=cfg,
            loader=loader,
            num_classes=len(class_ids),
            class_ids=class_ids,
            epochs=cfg.num_epochs,
            device=device,
            show_batch_progress=run_cfg.show_batch_progress,
        )
        fold_results.append(fold_result)
        all_true.extend(fold_result.y_true)
        all_pred.extend(fold_result.y_pred)

        mlflow.log_metric("fold_test_loss", fold_result.test_loss, step=fold_idx)
        mlflow.log_metric("fold_accuracy", fold_result.accuracy, step=fold_idx)
        mlflow.log_metric("fold_f1_macro", fold_result.f1_macro, step=fold_idx)
        mlflow.log_metric(f"test_loss_{split.identifier}", fold_result.test_loss)
        mlflow.log_metric(f"accuracy_{split.identifier}", fold_result.accuracy)
        mlflow.log_metric(f"f1_macro_{split.identifier}", fold_result.f1_macro)

        if fold_result.f1_macro > best_f1:
            best_f1 = fold_result.f1_macro
            best_fold_id = split.identifier
            best_model_for_logging = copy.deepcopy(trained_model).cpu().eval()

        fold_cm_path = output_dir / f"confusion_matrix_{split.identifier}.png"
        plot_confusion(
            fold_result.y_true,
            fold_result.y_pred,
            class_ids,
            class_names,
            fold_cm_path,
        )
        per_fold_confusion_paths[split.identifier] = str(fold_cm_path)

        split_iter.set_postfix(
            fold=split.identifier,
            test_loss=f"{fold_result.test_loss:.4f}",
            acc=f"{fold_result.accuracy:.4f}",
            f1_macro=f"{fold_result.f1_macro:.4f}",
        )

    loss_plot_path = output_dir / "loss_curve.png"
    plot_loss_curves(fold_results, loss_plot_path)

    cm_plot_path = output_dir / "confusion_matrix.png"
    plot_confusion(all_true, all_pred, class_ids, class_names, cm_plot_path)

    metrics = {
        "dataset": cfg.dataset_id,
        "training_variant": "balanced_loader_sampling",
        "k_folds": cfg.num_folds,
        "epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "seed": cfg.seed,
        "device": str(device),
        "folds": [
            {
                "fold": r.fold_id,
                "test_loss": r.test_loss,
                "accuracy": r.accuracy,
                "f1_macro": r.f1_macro,
            }
            for r in fold_results
        ],
        "mean_accuracy": float(np.mean([r.accuracy for r in fold_results])),
        "mean_f1_macro": float(np.mean([r.f1_macro for r in fold_results])),
        "per_fold_confusion_matrices": per_fold_confusion_paths,
        "loss_curve": str(loss_plot_path),
        "confusion_matrix": str(cm_plot_path),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    mlflow.log_metric("mean_accuracy", metrics["mean_accuracy"])
    mlflow.log_metric("mean_f1_macro", metrics["mean_f1_macro"])
    if best_fold_id is not None:
        mlflow.log_param("best_fold", best_fold_id)
        mlflow.log_param("best_fold_selection_metric", "f1_macro")

    if best_model_for_logging is not None:
        mlflow.pytorch.log_model(best_model_for_logging, name="best_model")

    mlflow.log_artifact(str(loss_plot_path), artifact_path="plots")
    mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")
    for fold_id, cm_path in per_fold_confusion_paths.items():
        mlflow.log_artifact(str(cm_path), artifact_path=f"plots/{fold_id}")
    mlflow.log_artifact(str(metrics_path), artifact_path="reports")
    mlflow.end_run()

    print(f"Saved loss curve: {loss_plot_path}")
    print(f"Saved confusion matrix: {cm_plot_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
