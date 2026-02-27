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

    # Model & trainer
    net = TUNet(
        input_channels=3,
        initial_channels=initial_channels,
        levels=levels,
        upsampling_method=upsampling_method,
        num_residual_layers=num_residual_layers,
        num_classes=3,
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
    MAX_FLOPS = 1
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

        # Reset the generator to ensure identical data sampling across trials for fair comparison
        path.p_data.reset_generator()

        # Train and get validation loss
        try:
            run_id, val_loss = trainer.train(
                num_epochs=1000,
                device=device,
                lr=lr,
                batch_size=128,
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
    # Set seeds for reproducibility
    torch.manual_seed(42)

    mlflow.set_experiment("standard_tunet")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=50,
            interval_steps=10,
            n_min_trials=5,
        ),
    )
    study.optimize(objective, n_trials=150)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    mlflow.set_experiment("best_models_retrained")

    path.p_data.reset_generator()  # Reset generator before retraining best model

    with mlflow.start_run(run_name="standard_unet") as run:
        run_id = run.info.run_id

        mlflow.log_params(study.best_params, run_id=run_id)

        # Retrain best model on full training data and evaluate metrics
        model = TUNet(
            input_channels=3,
            initial_channels=study.best_params["initial_channels"],
            levels=study.best_params["levels"],
            upsampling_method=study.best_params["upsampling_method"],
            num_residual_layers=study.best_params["num_residual_layers"],
            num_classes=3,
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
            num_epochs=1000,
            device=device,
            lr=study.best_params["learning_rate"],
            batch_size=128,
        )
        # Log the best model
        mlflow.pytorch.log_model(model, name="best_tunet", run_id=run_id)

        # Log final validation loss
        mlflow.log_metric("final_val_loss", val_loss, run_id=run_id)

        # Generate samples for evaluation
        with torch.no_grad():
            guidance_scales = [2.0, 3.0, 4.0]
            guidance_real_data = []
            guidance_generated_data = []

            # Sample real data once for all guidance scales
            real_data_all_classes = []
            for class_idx in range(1, 4):
                real_sensor_data, _ = path.p_data.sample(10000, class_idx=class_idx)
                real_data_all_classes.append(real_sensor_data)

            # Append the real data for all classes as a single entry in the guidance_real_data list
            guidance_real_data.append(real_data_all_classes)

            # Generate samples for each guidance scale
            for guidance_scale in guidance_scales:
                generated_per_scale = [
                    model.sample(
                        10000,
                        p_data_shape=[3, 128],
                        class_idx=class_idx,
                        guidance_scale=guidance_scale,
                    )
                    for class_idx in range(1, 4)
                ]
                guidance_generated_data.append(generated_per_scale)

        # Compute metrics
        metrics = compute_all_metrics(
            real_data=guidance_real_data[0],
            generated_data=guidance_generated_data,
            used_guidance_scales=guidance_scales,
            use_toy=True,
        )

        # Log metrics
        mlflow.log_metrics(metrics, run_id=run_id)
