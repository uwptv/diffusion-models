import mlflow
import optuna
import torch
from optuna.pruners import MedianPruner

from diffusion_models.architectures.standard_unet import StandardUNet
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.training_config import (
    TrainingConfig,
    run_objective_trial,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = TrainingConfig(
    dataset=WaveSampler(),
    num_classes=3,
    channels=3,
    sequence_length=128,
    experiment_name="standard_unet",
    model_name="standard_unet_toy",
    use_toy=True,
    num_trials=1,
    max_num_epochs=1,
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
        "num_residual_layers": trial.suggest_int("num_residual_layers", 1, 2),
        "cond_dim": trial.suggest_categorical("cond_dim", [48, 64]),
        "upsampling_method": trial.suggest_categorical(
            "upsampling_method", ["transposed", "interpolation", "pixel_shuffle"]
        ),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3]),
    }


def build_model(hyperparams: dict) -> StandardUNet:
    return StandardUNet(
        input_channels=cfg.channels,
        num_classes=cfg.num_classes,
        initial_channels=hyperparams["initial_channels"],
        levels=hyperparams["levels"],
        upsampling_method=hyperparams["upsampling_method"],
        num_residual_layers=hyperparams["num_residual_layers"],
        cond_dim=hyperparams["cond_dim"],
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

    # model, run_id = retrain_best_model(
    #     study=study,
    #     cfg=cfg,
    #     build_model=build_model,
    #     path=path,
    #     device=device,
    # )

    # # Evaluate the model on all metrics
    # evaluate_model(model, path, cfg, run_id)
