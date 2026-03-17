import torch

from diffusion_models.architectures.tiny_har import TinyHAR
from diffusion_models.data.loaders import DataSampler
from diffusion_models.dynamics.prob_paths import GaussianConditionalProbabilityPath
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.trainers import TinyHARTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create path
path = GaussianConditionalProbabilityPath(
    p_data=DataSampler(dataset="uci_har"),
    p_simple_shape=[3, 128],
    alpha=LinearAlpha(),
    beta=LinearBeta(),
).to(device)

# Create model
model = TinyHAR(input_channels=3, window_size=128, num_classes=6)

# Create trainer
trainer = TinyHARTrainer(
    model=model,
    path=path,
)

# Train
trainer.train(
    num_epochs=1000,
    device=device,
    name="tiny_har_uci_har",
    lr=1e-3,
    save_model=True,
    batch_size=128,
    class_names=[
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ],
)
