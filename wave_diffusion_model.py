import torch
from probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from distributions import WaveSampler
from utility import visualize_wave_path, visualize_generated_waves
from trainers import WaveTrainer
from backbones import TUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = WaveSampler(),
    p_simple_shape = [3, 100 * int(2 * torch.pi)],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# visualize_wave_path()

# initialize model
tunet = TUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
    input_channels = 3
    )

trainer = WaveTrainer(path = path, model = tunet, eta=0.1)
trainer.train(num_epochs = 1000, device=device, lr=1e-3, batch_size=250)

# visualize_generated_waves(model=tunet, guidance_scales=(1.0, 3.0, 5.0))
# does not seem to work that well :(