import mlflow
import optuna
import torch

from diffusion_models.architectures.tfilm_unet import TFiLMUNetTransformer
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

# activity_path = GaussianConditionalProbabilityPath(
#     p_data=DataSampler(dataset="wisdm"),
#     p_simple_shape=[3, 100],
#     alpha=LinearAlpha(),
#     beta=LinearBeta(),
# ).to(device)

# visualize generated waves
# visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))
# visualize_generated_data_samples(model=net, guidance_scales=(1.0, 2.0, 4.0))


# model is large at about 17.4 MiB parameters, trains resonably fast at about 7.7 it/s, empirically samples look pretty good after 1000 epochs, loss is about 0.45 after 1000 epochs


def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space
    initial_channels = trial.suggest_categorical("initial_channels", [16, 32, 48, 64])
    levels = trial.suggest_int("levels", 2, 4)
    num_residual_layers = trial.suggest_int("num_residual_layers", 1, 3)
    cond_dim = trial.suggest_categorical("cond_dim", [32, 64, 96, 128])
    eta = trial.suggest_float("label_dropout_rate", 0.05, 0.25, step=0.05)
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_epochs = trial.suggest_int("num_epochs", 200, 1000)
    num_tfilm_blocks = trial.suggest_categorical("num_tfilm_blocks", [8, 16])
    num_transformer_heads = trial.suggest_categorical(
        "num_transformer_heads", [1, 2, 4, 8]
    )
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 1, 4)
    ffn_dim_multiplier = trial.suggest_categorical("ffn_dim_multiplier", [1, 2, 4])

    # Model & trainer
    net = TFiLMUNetTransformer(
        input_channels=3,
        initial_channels=initial_channels,
        levels=levels,
        num_residual_layers=num_residual_layers,
        num_classes=3,
        cond_dim=cond_dim,
        num_tfilm_blocks=num_tfilm_blocks,
        num_transformer_heads=num_transformer_heads,
        num_transformer_layers=num_transformer_layers,
        ffn_dim_multiplier=ffn_dim_multiplier,
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
    MAX_FLOPS = 10
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
                "model_size_MiB": f"{model_size:.2f}",
                "flops_giga": f"{giga_flops:.3f}",
                "initial_channels": initial_channels,
                "levels": levels,
                "num_residual_layers": num_residual_layers,
                "cond_dim": cond_dim,
                "label_dropout_rate": f"{eta:.2f}",
                "learning_rate": f"{lr:.3}",
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "num_tfilm_blocks": num_tfilm_blocks,
                "num_transformer_heads": num_transformer_heads,
                "num_transformer_layers": num_transformer_layers,
                "ffn_dim_multiplier": ffn_dim_multiplier,
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
    mlflow.set_experiment("transformer_tfilm_unet")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    mlflow.set_experiment("best_models_retrained")

    with mlflow.start_run(run_name="transformer_tfilm_unet") as run:
        run_id = run.info.run_id

        mlflow.log_params(study.best_params, run_id=run_id)

        # Retrain best model on full training data and evaluate metrics
        model = TFiLMUNetTransformer(
            input_channels=3,
            initial_channels=study.best_params["initial_channels"],
            levels=study.best_params["levels"],
            num_residual_layers=study.best_params["num_residual_layers"],
            num_classes=3,
            cond_dim=study.best_params["cond_dim"],
            num_tfilm_blocks=study.best_params["num_tfilm_blocks"],
            num_transformer_heads=study.best_params["num_transformer_heads"],
            num_transformer_layers=study.best_params["num_transformer_layers"],
            ffn_dim_multiplier=study.best_params["ffn_dim_multiplier"],
        )
        trainer = CFGTrainer(
            path=path,
            model=model,
            eta=study.best_params["label_dropout_rate"],
            null_label=0,
        )
        _, val_loss = trainer.train(
            num_epochs=study.best_params["num_epochs"],
            device=device,
            lr=study.best_params["learning_rate"],
            batch_size=study.best_params["batch_size"],
            val_split=0.2,
        )
        # Log the best model
        mlflow.pytorch.log_model(
            model, artifact_path="best_standard_tfilm_model", run_id=run_id
        )

        # Log final validation loss
        mlflow.log_param("final_val_loss", val_loss, run_id=run_id)

        # Generate samples for evaluation
        with torch.no_grad():
            real_sensor_data, real_labels = path.p_data.sample(10000)
            generated_sensor_data = model.sample(10000, p_data_shape=[3, 128])

        # Compute metrics
        metrics = compute_all_metrics(
            real_data=real_sensor_data,
            generated_data=generated_sensor_data,
            use_toy=True,
        )

        # Log metrics
        mlflow.log_metrics(metrics, run_id=run_id)
