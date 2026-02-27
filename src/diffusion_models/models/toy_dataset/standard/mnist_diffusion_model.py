import torch

from ...architectures.MNIST_UNet import MNISTUNet
from ..dynamics.probability_paths import GaussianConditionalProbabilityPath
from ...dynamics.schedules import LinearAlpha, LinearBeta
from ...data.loaders import MNISTSampler
from ...trainers import CFGTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Initialize probability path
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialize model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    cond_dim = 80,
    num_classes = 10,
)

# Initialize trainer
trainer = CFGTrainer(path = path, model = unet, eta=0.1)

# Train!
trainer.train(num_epochs = 1000, device=device, lr=1e-3, batch_size=250)

# visualize_generated_mnist_samples(path = path, model = unet)