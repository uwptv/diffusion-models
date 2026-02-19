import torch

from diffusion_models.architectures.tiny_har import TinyHAR
from diffusion_models.data.synthetic import WaveSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import TinyHARTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data=WaveSampler(),
    p_simple_shape=[3, 128],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# activity_path = GaussianConditionalProbabilityPath(
#     p_data=DataSampler(dataset="wisdm"),
#     p_simple_shape=[3, 100],
#     alpha=LinearAlpha(),
#     beta=LinearBeta(),
# ).to(device)

# initialize model
net = TinyHAR(
    input_channels=3,
    window_size=128,
    num_classes=3,
)

trainer = TinyHARTrainer(path=path, model=net)
trainer.train(
    num_epochs=1000,
    device=device,
    lr=1e-3,
    batch_size=250,
    val_split=0.2,
    name="tiny_har_toy",
    save_path="checkpoints/tiny_har_toy.pth",
    confusion_matrix_samples=2000,
    class_names=["amp1", "amp2", "amp3"],
)
