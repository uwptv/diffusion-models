import torch

from diffusion_models.architectures.tunet import TUNetSeperable
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

# initialize model
net = TUNetSeperable(
    channels=[32, 64, 128],
    num_residual_layers=2,
    cond_dim=64,
    num_classes=3,
    num_t_blocks=4,
)

trainer = CFGTrainer(path=path, model=net, eta=0.1, null_label=0)
trainer.train(num_epochs=1000, device=device, lr=1e-3, batch_size=250)

# visualize generated waves
visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))

# model parameters: 18.5 MiB
# trains slowly at around 3.3 it/s
# loss at around 0.3 after 1000 epochs
# generated waves look ok (a little noisy)
