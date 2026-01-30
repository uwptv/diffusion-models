import torch

from diffusion_models.architectures.standard_unet import StandardUNet
from diffusion_models.data.synthetic import SineWaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils.visualizations import visualize_generated_sine_waves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=SineWaveSampler(),
    p_simple_shape=[1, 32, 32],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# Initialize model
unet = StandardUNet(
    channels=[32, 64, 128],
    num_residual_layers=2,
    cond_dim=64,
    num_classes=3,
)

# Initialize trainer
trainer = CFGTrainer(path=path, model=unet, eta=0.1, null_label=0)

# Train!
trainer.train(num_epochs=1000, device=device, lr=1e-3, batch_size=250)

visualize_generated_sine_waves(model=unet, guidance_scales=(1.0, 2.0, 4.0))

# Model summary:
# trains fast (about 10 it/s), has about 1.8 MiB parameters, achieves loss of around 0.379 after 1000 epochs, samples look good empirically.
