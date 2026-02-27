import torch

from diffusion_models.architectures.tiny_har import TinyHAR
from diffusion_models.data.loaders import DataSampler
from diffusion_models.trainers import TinyHARTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create samplers
train_sampler = DataSampler(dataset="wisdm", split_type="train")
val_sampler = DataSampler(dataset="wisdm", split_type="val")

# Create model
model = TinyHAR(input_channels=3, window_size=120, num_classes=6)

# Create trainer
trainer = TinyHARTrainer(
    model=model,
    train_sampler=train_sampler,
    val_sampler=val_sampler,
)

# Train
trainer.train(
    num_epochs=1000,
    device=device,
    name="tiny_har_wisdm",
    lr=1e-3,
    save_model=True,
    batch_size=128,
    class_names=["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"],
)
