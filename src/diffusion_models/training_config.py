import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import mlflow
import optuna
import torch

from diffusion_models.dynamics.prob_paths import Sampleable
from diffusion_models.metrics.evaluate_metrics import compute_all_metrics
from diffusion_models.trainers import CFGTrainer, EarlyStopping
from diffusion_models.utils.sizes import (
    GigaFLOP,
    MiB,
    count_flops,
    model_size_b,
    seed_everything,
)


@dataclass(frozen=True)
class TrainingConfig:
    # Dataset parameters
    dataset: Sampleable
    num_classes: int
    channels: int
    sequence_length: int
    experiment_name: str
    model_name: str
    evaluator: str

    # Training hyperparameters
    seed: int = 42
    batch_size: int = 128
    max_num_epochs: int = 1000
    patience: int = 75

    # Model constraints
    max_model_size_mib: float = 20.0
    max_gflops: float = 1.0

    # Optuna study parameters
    num_trials: int = 100
    num_startup_trials: int = 10
    n_warmup_steps: int = 50
    interval_steps: int = 10
    n_min_trials: int = 5

    # Evaluation parameters
    guidance_scales: list[float] = (2.0, 4.0)


def stable_params_key(params: dict) -> str:
    # deterministic across Python runs (unlike built-in hash)
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_objective_trial(
    trial: optuna.Trial,
    *,
    cfg: TrainingConfig,
    seen_configs: set[str],
    suggest_params: Callable[[optuna.Trial], dict],
    build_model: Callable[[dict], torch.nn.Module],
    path,
    device: torch.device,
) -> float:
    trainer = None
    model = None

    # Avoid cross-trial CUDA fragmentation/state buildup when running many Optuna trials.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    params = suggest_params(trial)
    params_key = stable_params_key(params)
    if params_key in seen_configs:
        raise optuna.TrialPruned("Already evaluated this configuration")
    seen_configs.add(params_key)

    seed_everything(cfg.seed)

    model = build_model(params)
    lr = params["learning_rate"]

    trainer = CFGTrainer(
        path=path,
        model=model,
        eta=0.1,
        null_class=cfg.num_classes,
        trial=trial,
        stopper=EarlyStopping(patience=cfg.patience),
    )

    model_size = model_size_b(model) / MiB
    if model_size > cfg.max_model_size_mib:
        with mlflow.start_run(
            run_name=f"trial_{trial.number}_pruned_size", nested=True
        ):
            mlflow.log_params(params)
            mlflow.log_param("model_size_MiB", model_size)
            mlflow.set_tag("status", "pruned_due_to_size")
        raise optuna.TrialPruned(f"Model too large: {model_size:.3f} MiB")

    giga_flops = (
        count_flops(model, channels=cfg.channels, seq_len=cfg.sequence_length)
        / GigaFLOP
    )
    if giga_flops > cfg.max_gflops:
        with mlflow.start_run(
            run_name=f"trial_{trial.number}_pruned_flops", nested=True
        ):
            mlflow.log_params(params)
            mlflow.log_param("giga_flops", giga_flops)
            mlflow.set_tag("status", "pruned_due_to_flops")
        raise optuna.TrialPruned(f"Model too large: {giga_flops:.3f} GFLOPs")

    try:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_size_MiB", f"{model_size:.2f}")
            mlflow.log_param("flops_giga", f"{giga_flops:.5f}")

            try:
                if hasattr(path.p_data, "reset_generator"):
                    path.p_data.reset_generator()

                run_id, val_loss = trainer.train(
                    num_epochs=cfg.max_num_epochs,
                    device=device,
                    lr=lr,
                    batch_size=cfg.batch_size,
                )
                mlflow.log_metric("val_loss", val_loss, run_id=run_id)
                return val_loss
            except optuna.TrialPruned:
                mlflow.set_tag("status", "pruned_during_training")
                mlflow.end_run(status="KILLED")
                raise optuna.TrialPruned(
                    "Trial pruned during training due to early stopping"
                )
            except Exception as e:
                mlflow.log_param("exception", str(e))
                mlflow.end_run(status="FAILED")
                raise e
    finally:
        # Explicitly release model/trainer references and cached CUDA memory between trials.
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def retrain_best_model(
    *,
    study: optuna.Study,
    cfg: TrainingConfig,
    build_model: Callable[[dict], torch.nn.Module],
    path,
    device: torch.device,
) -> tuple[torch.nn.Module, float, str]:
    """
    Retrain the best model from a study.

    Returns:
        tuple of (model, val_loss, mlflow_run_id)
    """
    mlflow.set_experiment("models_retrained")

    with mlflow.start_run(run_name=cfg.model_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(study.best_params, run_id=run_id)

        # Set seeds for reproducibility
        seed_everything(cfg.seed)

        # Build and train best model
        model = build_model(study.best_params)

        trainer = CFGTrainer(
            path=path,
            model=model,
            eta=0.1,
            null_class=cfg.num_classes,
            stopper=EarlyStopping(patience=cfg.patience),
        )

        # Reset data generator for reproducibility (if applicable)
        if hasattr(path.p_data, "reset_generator"):
            path.p_data.reset_generator()

        _, val_loss = trainer.train(
            num_epochs=cfg.max_num_epochs,
            lr=study.best_params["learning_rate"],
            batch_size=cfg.batch_size,
            device=device,
        )

        # Log the best model
        model_info = mlflow.pytorch.log_model(model, name=cfg.model_name, run_id=run_id)

        # Register the best model as an MLflow model version
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name=cfg.model_name,
        )

        # Log final validation loss
        mlflow.log_metric("final_val_loss", val_loss, run_id=run_id)

        return model, run_id


def evaluate_model(
    model: torch.nn.Module,
    path,
    cfg: TrainingConfig,
    mlflow_run_id: str,
) -> dict:
    with mlflow.start_run(run_id=mlflow_run_id):
        # Compute and log metrics
        metrics = compute_all_metrics(
            model=model,
            path=path,
            num_classes=cfg.num_classes,
            guidance_scales=cfg.guidance_scales,
            evaluator=cfg.evaluator,
        )

        mlflow.log_metrics(metrics)

        return metrics
