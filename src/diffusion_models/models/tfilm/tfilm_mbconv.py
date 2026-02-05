import torch

from diffusion_models.architectures.tfilm_unet import TFiLMMBConvUNet
from diffusion_models.data.loaders import DataSampler
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import CFGTrainer
from diffusion_models.utils.visualizations import visualize_generated_data_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 100 * int(2 * torch.pi)],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

activity_path = GaussianConditionalProbabilityPath(
    p_data=DataSampler(
        dataset="wisdm",
        window_time=6.0,
    ),
    p_simple_shape=[3, 100],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# initialize model
net = TFiLMMBConvUNet(
    channels=[32, 64, 128],
    num_residual_layers=2,
    cond_dim=64,
    num_classes=6,
    input_channels=3,
    num_tfilm_blocks=4,
)

trainer = CFGTrainer(path=activity_path, model=net, eta=0.1, null_label=6)
trainer.train(
    num_epochs=1000, device=device, lr=1e-3, batch_size=250, name="tfilm_mbconv_wisdm"
)

# visualize generated waves
# visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))
visualize_generated_data_samples(
    model=net, guidance_scales=(1.0, 2.0, 4.0), name="tfilm_mbconv_wisdm"
)


# model is large at about 17.4 MiB parameters, trains resonably fast at about 7.7 it/s, empirically samples look pretty good after 1000 epochs, loss is about 0.45 after 1000 epochs
