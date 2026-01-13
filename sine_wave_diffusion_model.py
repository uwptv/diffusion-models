import torch
from probability_paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from distributions import SineWaveSampler
from backbones import TUNet
from trainers import SineWaveTrainer
from utility import visualize_generated_sine_waves, visualize_sine_wave_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize probability path for sine wave generation
path = GaussianConditionalProbabilityPath(
    p_data = SineWaveSampler(),
    p_simple_shape = [1, 100 * int(2 * torch.pi)],  # (channels = 1, sample_rate * duration)
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# visualize_sine_wave_path()

# Initialize model for sine wave generation
tunet = TUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    cond_dim=64,
    num_classes=3
)

trainer = SineWaveTrainer(path = path, model = tunet, eta=0.1)
trainer.train(num_epochs = 1000, device=device, lr=1e-3, batch_size=250)

visualize_generated_sine_waves(
    model=tunet, 
    samples_per_amplitude=3,
    guidance_scales=(1.0, 3.0, 5.0)
)