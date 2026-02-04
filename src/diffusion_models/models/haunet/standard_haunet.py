import torch

from diffusion_models.architectures.HAUNet import HAUNet
from diffusion_models.data.loaders import DataSampler
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils.visualizations import (
    visualize_generated_data_samples,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 100 * int(2 * torch.pi)],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

activity_path = GaussianConditionalProbabilityPath(
    p_data=DataSampler(dataset="wisdm", window_time=6.0),
    p_simple_shape=[3, 100],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# visualize_wave_path()

# initialize model
net = HAUNet(
    num_residual_layers=2,
    num_encoder_decoder_layers=3,
    cond_dim=64,
    num_classes=6,
)

trainer = CFGTrainer(path=activity_path, model=net, eta=0.1, null_label=6)
trainer.train(num_epochs=1000, device=device, lr=1e-3, batch_size=50, name="HAUNet")

# visualize_generated_waves(
#     model=net, guidance_scales=(1.0, 2.0, 4.0), name="HAUNet_generated_waves"
# )
visualize_generated_data_samples(
    model=net, guidance_scales=(1.0, 2.0, 4.0), name="HAUNet_generated_data_samples"
)

# model size: 7 MiB
# trains very slowly at around 1.1 it/s
# loss at around 0.5 with heavy fluctuations after 1000 epochs
# generated waves look
