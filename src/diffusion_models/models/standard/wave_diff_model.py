import mlflow
import optuna
import torch

from diffusion_models.architectures.standard_unet import StandardUNet
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.metrics.evaluate_metrics import compute_all_metrics
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils import MiB, model_size_b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 100],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)


def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space
    initial_channels = trial.suggest_categorical("initial_channels", [16, 32, 48, 64])
    levels = trial.suggest_int("levels", 2, 4)
    num_residual_layers = trial.suggest_int("num_residual_layers", 1, 3)
    cond_dim = trial.suggest_categorical("cond_dim", [32, 64, 96, 128])
    eta = trial.suggest_float("eta", 0.05, 0.5, log=True)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 250, 512])
    num_epochs = trial.suggest_int("num_epochs", 200, 1000)

    # Model & trainer
    net = StandardUNet(
        input_channels=3,
        initial_channels=initial_channels,
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_classes=3,
        cond_dim=cond_dim,
    )
    trainer = CFGTrainer(path=path, model=net, eta=eta, null_label=0)

    with mlflow.start_run(nested=True):
        model_size = model_size_b(net)
        mlflow.log_params(
            {
                "initial_channels": initial_channels,
                "levels": levels,
                "num_residual_layers": num_residual_layers,
                "cond_dim": cond_dim,
                "eta": eta,
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_size_MiB": model_size / MiB,
            }
        )

        # Train and get validation loss
        run_id, val_loss = trainer.train(
            num_epochs=num_epochs,
            device=device,
            lr=lr,
            batch_size=batch_size,
            val_split=0.2,  # Use 20% of data for validation
        )

        mlflow.log_metric("val_loss", val_loss, run_id=run_id)

        # Generate samples for evaluation
        with torch.no_grad():
            real_sensor_data, real_labels = path.p_data.sample(10000)
            generated_sensor_data = net.sample(10000, p_data_shape=[3, 100])

        # Compute metrics
        metrics = compute_all_metrics(
            real_data=real_sensor_data,
            generated_data=generated_sensor_data,
        )

        # Log metrics
        mlflow.log_metrics(metrics, run_id=run_id)

        # Optimize for validation loss (lower is better)
        return val_loss


if __name__ == "__main__":
    mlflow.set_experiment("standard_unet_optuna")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
