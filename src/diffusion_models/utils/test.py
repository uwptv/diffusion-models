import mlflow.pytorch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/standard_unet_toy/2").to(device)

model.visualize(
    p_data_shape=[3, 128],
    save_path="plots/signals/standard_unet_toy.png",
)
