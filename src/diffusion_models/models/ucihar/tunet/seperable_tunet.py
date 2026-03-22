import mlflow
import optuna
import torch
from optuna.pruners import MedianPruner

from diffusion_models.architectures.tunet import SeperableTUNet
from diffusion_models.data.loaders import DataSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.training_config import (
    TrainingConfig,
    evaluate_model,
    retrain_best_model,
    run_objective_trial,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = TrainingConfig(
    dataset=DataSampler(dataset="uci_har"),
    num_classes=6,
    channels=3,
    sequence_length=128,
    experiment_name="seperable_tunet_uci_har",
    model_name="seperable_tunet_uci_har",
    evaluator="ucihar",
    guidance_scales=[2.0],
)

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=cfg.dataset,
    p_simple_shape=[cfg.channels, cfg.sequence_length],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

seen_configs = set()


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "initial_channels": trial.suggest_categorical("initial_channels", [4, 8, 16]),
        "levels": trial.suggest_int("levels", 1, 2),
        "cond_dim": trial.suggest_categorical("cond_dim", [48, 64]),
        "upsampling_method": trial.suggest_categorical(
            "upsampling_method", ["transposed", "interpolation", "pixel_shuffle"]
        ),
        "num_residual_layers": trial.suggest_int("num_residual_layers", 1, 2),
        "num_tfilm_blocks": trial.suggest_categorical(
            "num_tfilm_blocks", [2, 4, 8, 16]
        ),
        "hidden_size_rnn": trial.suggest_categorical("hidden_size_rnn", [32, 64, 128]),
        "num_layers_rnn": trial.suggest_int("num_layers_rnn", 1, 2),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "ffn_expansion_factor": trial.suggest_categorical(
            "ffn_expansion_factor", [2, 4, 8]
        ),
        "filters_per_channel": trial.suggest_categorical(
            "filters_per_channel", [2, 4, 8]
        ),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3]),
    }


def build_model(hyperparams: dict) -> SeperableTUNet:
    return SeperableTUNet(
        input_channels=cfg.channels,
        num_classes=cfg.num_classes,
        initial_channels=hyperparams["initial_channels"],
        levels=hyperparams["levels"],
        upsampling_method=hyperparams["upsampling_method"],
        num_residual_layers=hyperparams["num_residual_layers"],
        cond_dim=hyperparams["cond_dim"],
        num_tfilm_blocks=hyperparams["num_tfilm_blocks"],
        hidden_size_rnn=hyperparams["hidden_size_rnn"],
        num_layers_rnn=hyperparams["num_layers_rnn"],
        num_heads=hyperparams["num_heads"],
        ffn_expansion_factor=hyperparams["ffn_expansion_factor"],
        filters_per_channel=hyperparams["filters_per_channel"],
    )


def objective(trial: optuna.Trial) -> float:
    return run_objective_trial(
        trial,
        cfg=cfg,
        seen_configs=seen_configs,
        suggest_params=suggest_params,
        build_model=build_model,
        path=path,
        device=device,
    )


if __name__ == "__main__":
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.set_experiment_tag("dataset", "uci_har")
    mlflow.set_experiment_tag("model_family", "seperable_tunet")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(
            n_startup_trials=cfg.num_startup_trials,
            n_warmup_steps=cfg.n_warmup_steps,
            interval_steps=cfg.interval_steps,
            n_min_trials=cfg.n_min_trials,
        ),
    )
    study.optimize(objective, n_trials=cfg.num_trials)

    model, run_id = retrain_best_model(
        study=study,
        cfg=cfg,
        build_model=build_model,
        path=path,
        device=device,
    )

    # Evaluate the model on all metrics
    evaluate_model(model, path, cfg, run_id)
