import mlflow.pytorch
import torch

from diffusion_models.data.synthetic import WaveSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/standard_unet_toy/4").to(device)
sampler = WaveSampler()
mean, std = sampler.get_mean_std()

model.visualize(
    p_data_shape=[3, 128],
    dataset_mean=mean.to(device),
    dataset_std=std.to(device),
    save_path="plots/signals/standard_unet_toy.png",
)

# t-SNE visualization
real_data = []
for class_idx in range(3):
    samples, _ = sampler.sample(num_samples=100, class_idx=class_idx, normalize=False)
    samples = sampler.denormalize(samples)
    real_data.append(samples)

# Generate t-SNE plot
model.plot_tsne(
    p_data_shape=[3, 128],
    real_data=real_data,
    dataset_mean=mean.to(device),
    dataset_std=std.to(device),
    num_samples=100,
    guidance_scale=2.0,
    class_names=["amp1", "amp2", "amp3"],
    device=device,
)
