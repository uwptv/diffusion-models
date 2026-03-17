import mlflow.pytorch
import torch

from diffusion_models.data.loaders import DataSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/standard_unet_wisdm@latest").to(device)
sampler = DataSampler()

# model.visualize(
#     p_data_shape=[3, 120],
#     dataset_mean=0.0,
#     dataset_std=1.0,
#     save_path="plots/signals/standard_unet_wisdm.png",
# )

# t-SNE visualization
real_data = []
for class_idx in range(6):
    samples, _ = sampler.sample(num_samples=100, class_idx=class_idx)
    real_data.append(samples)

# Generate t-SNE plot
model.plot_tsne(
    p_data_shape=[3, 120],
    real_data=real_data,
    num_samples=100,
    guidance_scale=2.0,
    class_names=["class1", "class2", "class3", "class4", "class5", "class6"],
    device=device,
)
