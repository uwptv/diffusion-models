import mlflow
import optuna
import torch

from diffusion_models.architectures.standard_unet import StandardUNet
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.metrics.evaluate_metrics import compute_all_metrics
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils.sizes import GigaFLOP, MiB, count_flops, model_size_b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 128],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)


def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space
    initial_channels = trial.suggest_categorical("initial_channels", [16, 32, 48, 64])
    levels = trial.suggest_int("levels", 2, 4)
    num_residual_layers = trial.suggest_int("num_residual_layers", 1, 3)
    cond_dim = trial.suggest_categorical("cond_dim", [32, 64, 96, 128])
    eta = trial.suggest_float("eta", 0.05, 0.25, step=0.05)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
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

    # Skip models that are too large to train
    model_size = model_size_b(net) / MiB
    MAX_MODEL_SIZE = 20
    if model_size > MAX_MODEL_SIZE:
        with mlflow.start_run(
            run_name=f"trial_{trial.number}_pruned_size", nested=True
        ):
            mlflow.log_params(trial.params)
            mlflow.set_tag("status", "pruned_due_to_size")
            mlflow.log_param("model_size_MiB", model_size)

        raise optuna.TrialPruned(f"Model too large: {model_size: .3f}MiB")

    # Skip models that have too many GFLOPs for training
    flops = count_flops(net, channels=3, seq_len=128)
    giga_flops = flops / GigaFLOP
    MAX_FLOPS = 100
    if giga_flops > MAX_FLOPS:
        with mlflow.start_run(
            run_name=f"trial_{trial.number}_pruned_flops", nested=True
        ):
            mlflow.log_params(trial.params)
            mlflow.set_tag("status", "pruned_due_to_flops")
            mlflow.log_param("giga_flops", giga_flops)

        raise optuna.TrialPruned(f"Model too large: {giga_flops: .3f} GFLOPs")

    # Train and log results
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(
            {
                "initial_channels": initial_channels,
                "levels": levels,
                "num_residual_layers": num_residual_layers,
                "cond_dim": cond_dim,
                "label_dropout_rate": f"{eta:.2f}",
                "learning_rate": f"{lr:.3}",
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "model_size_MiB": f"{model_size:.2f}",
                "flops_giga": f"{giga_flops:.3f}",
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

        # Optimize for validation loss (lower is better)
        return val_loss


if __name__ == "__main__":
    mlflow.set_experiment("standard_unet")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    with mlflow.start_run(run_name="best_model_retraining"):
        # Retrain best model on full training data and evaluate metrics
        model = StandardUNet(
            input_channels=3,
            initial_channels=study.best_params["initial_channels"],
            levels=study.best_params["levels"],
            num_residual_layers=study.best_params["num_residual_layers"],
            num_classes=3,
            cond_dim=study.best_params["cond_dim"],
        )
        trainer = CFGTrainer(
            path=path, model=model, eta=study.best_params["label_dropout_rate"]
        )
        run_id, _ = trainer.train(
            num_epochs=study.best_params["num_epochs"],
            device=device,
            lr=study.best_params["learning_rate"],
            batch_size=study.best_params["batch_size"],
            val_split=0.2,
        )
        # Log the best model
        mlflow.pytorch.log_model(model, artifact_path="best_model")
        mlflow.log_params(study.best_params, run_id=run_id)

    # Generate samples for evaluation
    with torch.no_grad():
        real_sensor_data, real_labels = path.p_data.sample(10000)
        generated_sensor_data = model.sample(10000, p_data_shape=[3, 128])

    # Compute metrics
    metrics = compute_all_metrics(
        real_data=real_sensor_data,
        generated_data=generated_sensor_data,
    )

    # Log metrics
    mlflow.log_metrics(metrics, run_id=run_id)
