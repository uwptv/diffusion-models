import torch
from backbones import TFiLMUNet
from probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from distributions import WaveSampler
from trainers import WaveTrainer
from utility import visualize_generated_waves

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

trainer = WaveTrainer(path = path, model = net, eta=0.1)
trainer.train(num_epochs = 2000, device=device, lr=1e-3, batch_size=250)

# visualize generated waves
visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))