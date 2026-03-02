import mlflow.pytorch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/BestTFiLMUNetToy/1").to(device)

model.visualize(
    num_samples=4,
    p_data_shape=[3, 128],
    num_timesteps=30,
    class_idx=2,
    guidance_scale=2.0,
    save_path="plots/signals/tfilm_unet_toy_3.png",
)
