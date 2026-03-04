import mlflow.pytorch
import torch

from diffusion_models.data.synthetic import WaveSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/standard_unet_toy/4").to(device)
sampler = WaveSampler()

# model.visualize(
#     p_data_shape=[3, 128],
#     save_path="plots/signals/standard_unet_toy.png",
# )

# t-SNE visualization
real_data = []
for class_idx in range(3):
    samples, _ = sampler.sample(num_samples=100, class_idx=class_idx)
    samples = sampler.denormalize(samples).to(device)
    real_data.append(samples.cpu())

# Generate t-SNE plot
model.plot_tsne(
    p_data_shape=[3, 128],
    real_data=real_data,
    num_samples=100,
    guidance_scale=2.0,
    class_names=["amp1", "amp2", "amp3"],
    device=device,
)
