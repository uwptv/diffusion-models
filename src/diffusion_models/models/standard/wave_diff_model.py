import mlflow
import torch

from diffusion_models.architectures.standard_unet import StandardUNet
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.metrics.kid import compute_kid_with_encoder
from diffusion_models.trainers import CFGTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 100],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# activity_path = GaussianConditionalProbabilityPath(
#     p_data=DataSampler(dataset="wisdm"),
#     p_simple_shape=[3, 100],
#     alpha=LinearAlpha(),
#     beta=LinearBeta(),
# ).to(device)

# visualize_gaussian_cond_prob_path(path=activity_path, num_samples=4, num_timesteps=5)

# initialize model
net = StandardUNet(
    channels=[
        32,
        64,
        128,
    ],
    num_residual_layers=2,
    cond_dim=64,
    num_classes=3,
    input_channels=3,
)

trainer = CFGTrainer(path=path, model=net, eta=0.1, null_label=0)
run_id = trainer.train(
    num_epochs=1000,
    device=device,
    name="standard_wave_diffusion_model",
    lr=1e-3,
    batch_size=250,
)

# visualize_generated_waves(model=net, guidance_scales=(1.0, 2.0, 4.0))
# visualize_generated_data_samples(model=net, guidance_scales=(1.0, 2.0, 4.0))

# model size is 2.6MiB parameters, trains resonably fast at at about 8.4 it/s, loss is about 1.3 after 1000 epochs, samples dont look very good empirically but one can see some structure

real_sensor_data, real_labels = path.p_data.sample(1000)
generated_sensor_data = net.sample(1000, p_data_shape=[3, 100])

# Compute the KID metric
kid_dict = compute_kid_with_encoder(
    real_data=real_sensor_data,
    generated_data=generated_sensor_data,
    batch_size=250,
)

mlflow.log_metrics(kid_dict, run_id=run_id)
