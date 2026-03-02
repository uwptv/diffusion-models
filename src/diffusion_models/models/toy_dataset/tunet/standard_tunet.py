import mlflow
import optuna
import torch
from optuna.pruners import MedianPruner

from diffusion_models.architectures.tunet import TUNet
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.metrics.evaluate_metrics import compute_all_metrics
from diffusion_models.trainers import CFGTrainer, EarlyStopping
from diffusion_models.utils.sizes import (
    GigaFLOP,
    MiB,
    count_flops,
    model_size_b,
    seed_everything,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set training depending constants
NUM_CLASSES = 3
USE_TOY = True

# Set global constants
SEED = 42
MAX_MODEL_SIZE = 20
MAX_GFLOPS = 1
BATCH_SIZE = 128
MAX_NUM_EPOCHS = 1000

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 128],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

seen_configs = set()


def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space for model architecture
    initial_channels = trial.suggest_categorical("initial_channels", [4, 8, 16])
    levels = trial.suggest_int("levels", 1, 2)
    num_residual_layers = trial.suggest_int("num_residual_layers", 1, 2)
    cond_dim = trial.suggest_categorical("cond_dim", [48, 64])
    upsampling_method = trial.suggest_categorical(
        "upsampling_method", ["transposed", "interpolation", "pixel_shuffle"]
    )
    num_tfilm_blocks = trial.suggest_categorical("num_tfilm_blocks", [2, 4, 8, 16])
    hidden_size_rnn = trial.suggest_categorical("hidden_size_rnn", [32, 64, 128])
    num_layers_rnn = trial.suggest_int("num_layers_rnn", 1, 3)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 1, 3)
    ffn_expansion_factor = trial.suggest_categorical("ffn_expansion_factor", [2, 4, 8])

    # Hyperparameters for training
    eta = trial.suggest_categorical("label_dropout_rate", [0.1, 0.2])
    lr = trial.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3])

    # Get a hash of the hyperparameters to avoid retraining the same model multiple times
    params_hash = hash(
        (
            initial_channels,
            levels,
            num_residual_layers,
            cond_dim,
            upsampling_method,
            eta,
            lr,
        )
    )

    if params_hash in seen_configs:
        raise optuna.TrialPruned("Already evaluated this configuration")
    seen_configs.add(params_hash)

    # Reset seeds for reproducibility in each trial
    seed_everything()

    # Model & trainer
    net = TUNet(
        input_channels=3,
        initial_channels=initial_channels,
        levels=levels,
        upsampling_method=upsampling_method,
        num_residual_layers=num_residual_layers,
        num_classes=NUM_CLASSES,
        cond_dim=cond_dim,
        num_tfilm_blocks=num_tfilm_blocks,
        hidden_size_rnn=hidden_size_rnn,
        num_layers_rnn=num_layers_rnn,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers,
        ffn_expansion_factor=ffn_expansion_factor,
    )
    trainer = CFGTrainer(
        path=path, model=net, eta=eta, trial=trial, stopper=EarlyStopping(patience=50)
    )

    # Skip models that are too large to train
    model_size = model_size_b(net) / MiB
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
    if giga_flops > MAX_GFLOPS:
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
                "flops_giga": f"{giga_flops:.5f}",
                "initial_channels": initial_channels,
                "levels": levels,
                "upsampling_method": upsampling_method,
                "num_residual_layers": num_residual_layers,
                "cond_dim": cond_dim,
                "num_tfilm_blocks": num_tfilm_blocks,
                "hidden_size_rnn": hidden_size_rnn,
                "num_layers_rnn": num_layers_rnn,
                "num_heads": num_heads,
                "num_transformer_layers": num_transformer_layers,
                "ffn_expansion_factor": ffn_expansion_factor,
                "label_dropout_rate": f"{eta:.2f}",
                "learning_rate": f"{lr:.3}",
            }
        )

        # Train and get validation loss
        try:
            run_id, val_loss = trainer.train(
                num_epochs=MAX_NUM_EPOCHS,
                device=device,
                lr=lr,
                batch_size=BATCH_SIZE,
            )

            mlflow.log_metric("val_loss", val_loss, run_id=run_id)

            # Optimize for validation loss (lower is better)
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


if __name__ == "__main__":
    mlflow.set_experiment("standard_tunet")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=50,
            interval_steps=10,
            n_min_trials=5,
        ),
    )
    study.optimize(objective, n_trials=100)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    mlflow.set_experiment("best_models_retrained")

    with mlflow.start_run(run_name="standard_unet") as run:
        run_id = run.info.run_id

        mlflow.log_params(study.best_params, run_id=run_id)

        seed_everything()

        # Retrain best model on full training data and evaluate metrics
        model = TUNet(
            input_channels=3,
            initial_channels=study.best_params["initial_channels"],
            levels=study.best_params["levels"],
            upsampling_method=study.best_params["upsampling_method"],
            num_residual_layers=study.best_params["num_residual_layers"],
            num_classes=NUM_CLASSES,
            cond_dim=study.best_params["cond_dim"],
            num_tfilm_blocks=study.best_params["num_tfilm_blocks"],
            hidden_size_rnn=study.best_params["hidden_size_rnn"],
            num_layers_rnn=study.best_params["num_layers_rnn"],
            num_heads=study.best_params["num_heads"],
            num_transformer_layers=study.best_params["num_transformer_layers"],
            ffn_expansion_factor=study.best_params["ffn_expansion_factor"],
        )
        trainer = CFGTrainer(
            path=path,
            model=model,
            eta=study.best_params["label_dropout_rate"],
            stopper=EarlyStopping(patience=50),
        )
        _, val_loss = trainer.train(
            num_epochs=MAX_NUM_EPOCHS,
            device=device,
            lr=study.best_params["learning_rate"],
            batch_size=BATCH_SIZE,
        )
        # Log the best model
        model_info = mlflow.pytorch.log_model(model, name="best_tunet", run_id=run_id)

        # Register the best model as an MLflow model version
        mlflow.register_model(
            model_uri=model_info.model_uri,
            name="BestTUNetToy",
        )

        # Log final validation loss
        mlflow.log_metric("final_val_loss", val_loss, run_id=run_id)

        metrics = compute_all_metrics(model, path, NUM_CLASSES, [2.0, 4.0], USE_TOY)

        # Log metrics
        mlflow.log_metrics(metrics, run_id=run_id)
