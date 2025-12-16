import matplotlib.pyplot as plt
import torch
from torch import nn
from distributions import MNISTSampler, SineWaveSampler
from probability_paths import ConditionalProbabilityPath, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from differential_equations import ConditionalVectorField, CFGVectorFieldODE
from simulators import EulerSimulator
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )

MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

def visualize_gaussian_cond_prob_path():
    num_rows = 10
    num_cols = 10
    num_timesteps = 5

    # Initialize our sampler
    sampler = MNISTSampler().to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = MNISTSampler(),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Sample 
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, 32, 32)

    # Setup plot
    fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1) # (num_samples, 1, 1, 1)
        xt = path.sample_conditional_path(z, tt) # (num_samples, 1, 32, 32)
        grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
        axes[tidx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[tidx].axis("off")
    plt.show()

def visualize_sine_wave_path():
    num_rows = 5
    num_cols = 5
    num_timesteps = 5

    # Initialize our sampler
    sampler = SineWaveSampler().to(device)

    # Initialize probability path
    path = GaussianConditionalProbabilityPath(
        p_data = sampler,
        p_simple_shape = [int(100.0 * 1.0)],  # sample_rate * duration
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Sample 
    num_samples = num_rows * num_cols
    z, _ = path.p_data.sample(num_samples)
    z = z.view(-1, 1, int(100.0 * 1.0))  # (num_samples, 1, signal_length)

    # Setup plot
    fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

    # Sample from conditional probability paths and graph
    ts = torch.linspace(0, 1, num_timesteps).to(device)
    for tidx, t in enumerate(ts):
        tt = t.view(1,1,1).expand(num_samples, 1, 1) # (num_samples, 1, 1)
        xt = path.sample_conditional_path(z, tt) # (num_samples, 1, signal_length)
        
        for i in range(num_samples):
            axes[tidx].plot(xt[i,0].cpu(), alpha=0.5)
        axes[tidx].set_title(f'Time t={t.item():.2f}')
    plt.show()

def visualize_generated_mnist_samples(path: ConditionalProbabilityPath, model: ConditionalVectorField):
    samples_per_class = 10
    num_timesteps = 100
    guidance_scales = [1.0, 3.0, 5.0]

    # Graph
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(model, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
        num_samples = y.shape[0]
        x0, _ = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

        # Simulate
        ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)

        # Plot
        grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
    plt.show()

def visualize_sinewave_samples(samples: torch.Tensor):
    num_samples = samples.shape[0]
    t = torch.linspace(0, 1, samples.shape[1])

    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.plot(t.cpu(), samples[i].cpu(), label=f'Sample {i+1}')
    plt.title('Sine Wave Samples')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()