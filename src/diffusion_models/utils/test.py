import mlflow.pytorch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/Standard_UNet_wisdm/1").to(device)

model.visualize(
    num_samples=4,
    p_data_shape=[3, 128],
    class_idx=3,
    num_timesteps=30,
    guidance_scale=2.0,
    class_names=["Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs"],
    save_path="plots/standard_unet_wisdm_visualization.png",
)
