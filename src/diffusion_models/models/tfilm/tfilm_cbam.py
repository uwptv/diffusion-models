import torch

from diffusion_models.architectures.tfilm_unet import TFiLMUNetCBAM
from diffusion_models.data.loaders import DataSampler
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils.visualizations import visualize_generated_waves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 100 * int(2 * torch.pi)],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

activity_path = GaussianConditionalProbabilityPath(
    p_data=DataSampler(dataset="wisdm"),
    p_simple_shape=[3, 100],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# initialize model
net = TFiLMUNetCBAM(
    channels=[32, 64, 128],
    num_residual_layers=2,
    cond_dim=64,
    num_classes=6,
    input_channels=3,
    num_tfilm_blocks=4,
)

trainer = CFGTrainer(path=path, model=net, eta=0.1, null_label=6)
trainer.train(num_epochs=1000, device=device, lr=1e-3, batch_size=250)

# visualize generated waves
visualize_generated_waves(
    model=net, name="TFiLMUNetCBAM", guidance_scales=(1.0, 2.0, 4.0)
)
# visualize_generated_data_samples(model=net, guidance_scales=(1.0, 2.0, 4.0))

# model parameters: 14.1 MiB
# trains fast at about 7.5 it/s
# loss is about 0.35 after 1000 epochs
# samples look ok empirically, pretty noisy on lower amplitudes but cross-channel correlations look good
