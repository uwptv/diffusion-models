import torch

from diffusion_models.architectures.tfilm_unet import TFiLMUNet
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.trainers import CFGTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = WaveSampler(),
    p_simple_shape = [3, 100 * int(2 * torch.pi)],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# initialize model
net = TFiLMUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    cond_dim=64,
    num_classes=3,
    input_channels=3,
    num_tfilm_blocks=4
    )

trainer = CFGTrainer(path = path, model = net, eta=0.1, null_label=0)
trainer.train(num_epochs = 2000, device=device, lr=1e-3, batch_size=250)

# visualize generated waves
# visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))