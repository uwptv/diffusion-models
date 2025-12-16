import torch
from probability_paths import ConditionalProbabilityPath, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from distributions import MNISTSampler, SineWaveSampler
from backbones import MNISTUNet, SineWaveUNet
from trainers import CFGTrainer
from utility import visualize_generated_mnist_samples, visualize_sine_wave_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# Initialize probability path
# path = GaussianConditionalProbabilityPath(
#     p_data = MNISTSampler(),
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)

# Initialize model
# unet = MNISTUNet(
#     channels = [32, 64, 128],
#     num_residual_layers = 2,
#     t_embed_dim = 40,
#     y_embed_dim = 40,
# )

# Initialize trainer
# trainer = CFGTrainer(path = path, model = unet, eta=0.1)

# Train!
# trainer.train(num_epochs = 1000, device=device, lr=1e-3, batch_size=250)

# visualize_generated_mnist_samples(path = path, model = unet)

#Initialize probability path for sine wave generation
path = GaussianConditionalProbabilityPath(
    p_data = SineWaveSampler(),
    p_simple_shape = [int(100.0 * 1.0)],  # sample_rate * duration
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

visualize_sine_wave_path()

# Initialize model for sine wave generation
tunet = SineWaveUNet(
    channels = [64, 128, 256],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
)

trainer = CFGTrainer(path = path, model = tunet, eta=0.1)
trainer.train(num_epochs = 500, device=device, lr=1e-3, batch_size=250)
