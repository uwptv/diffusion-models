import mlflow.pytorch
import torch

from diffusion_models.data.loaders import DataSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get model from mlflow
model = mlflow.pytorch.load_model("models:/tfilm_unet_uci_har@latest").to(device)
sampler = DataSampler(dataset="uci_har")

model.visualize(
    p_data_shape=[9, 128],
    guidance_scales=[2.0, 3.0, 4.0],
    dataset_mean=0.0,
    dataset_std=1.0,
    save_path="plots/signals/tfilm_unet_uci_har.png",
    class_names=[
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    ],
)

# t-SNE visualization
# real_data = []
# for class_idx in range(6):
#     samples, _ = sampler.sample(num_samples=100, class_idx=class_idx)
#     real_data.append(samples)

# # Generate t-SNE plot
# model.plot_tsne(
#     p_data_shape=[9, 128],
#     real_data=real_data,
#     num_samples=100,
#     perplexity=40,
#     guidance_scale=3.0,
#     class_names=[
#         "walking",
#         "walking_upstairs",
#         "walking_downstairs",
#         "sitting",
#         "standing",
#         "laying",
#     ],
#     device=device,
# )
