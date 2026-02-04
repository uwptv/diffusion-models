import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion_models.data.loaders import DataSampler
from diffusion_models.data.synthetic import SineWaveSampler, WaveSampler
from diffusion_models.dynamics.base import CFGVectorFieldODE, ConditionalVectorField
from diffusion_models.dynamics.prob_paths import (
    ConditionalProbabilityPath,
    GaussianConditionalProbabilityPath,
)
from diffusion_models.dynamics.schedules import LinearAlpha, LinearBeta
from diffusion_models.dynamics.simulators import EulerSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_gaussian_cond_prob_path(
    path: ConditionalProbabilityPath,
    num_samples: int = 12,
    num_timesteps: int = 5,
):
    """
    Visualize a conditional probability path by showing samples at different timesteps.
    Similar to visualize_sine_wave_path, groups samples by their labels.

    Args:
        - path: a ConditionalProbabilityPath instance to visualize
        - num_samples: number of samples to visualize
        - num_timesteps: number of timesteps to visualize
    """
    # Sample conditioning variables and labels
    z, labels = path.p_data.sample(num_samples)

    # Get unique classes and group samples by class
    unique_classes = torch.unique(labels).cpu()
    num_classes = len(unique_classes)

    ts = torch.linspace(0, 1, num_timesteps, device=device)

    # Create subplots: one row per class
    fig, axes = plt.subplots(
        num_classes,
        num_timesteps,
        figsize=(3 * num_timesteps, 2 * num_classes),
        sharex=True,
        sharey=True,
    )

    # Handle case where there's only one class
    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_label in enumerate(unique_classes):
        # Get all samples with this class label
        mask = labels.squeeze() == class_label
        class_z = z[mask]
        num_class_samples = class_z.shape[0]

        for tidx, t in enumerate(ts):
            tt = t.expand(num_class_samples, 1, 1)  # shape (num_class_samples, 1, 1)
            xt = (
                path.sample_conditional_path(class_z, tt).detach().cpu()
            )  # (num_class_samples, ...)
            xt = xt.reshape(
                num_class_samples, -1
            )  # Reshape to (num_class_samples, features)

            ax = axes[class_idx, tidx]

            # Plot all samples in this class
            for sidx in range(num_class_samples):
                ax.plot(xt[sidx], alpha=0.7)

            if class_idx == 0:
                ax.set_title(f"t={float(t):.2f}", fontsize=10)
            if tidx == 0:
                ax.set_ylabel(
                    f"Class={int(class_label)}", fontsize=10, fontweight="bold"
                )
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Gaussian conditional path (grouped by class)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def visualize_sine_wave_path():
    num_samples = 12
    num_timesteps = 5

    sampler = SineWaveSampler()
    signal_length = sampler.sample_rate * sampler.duration
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[1, signal_length],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    z, labels = path.p_data.sample(
        num_samples
    )  # z shape (num_samples, 1, signal_len), labels shape (num_samples, 1)

    # Get unique amplitude classes and group samples by class
    unique_amplitudes = torch.unique(labels).cpu()
    num_classes = len(unique_amplitudes)

    ts = torch.linspace(0, 1, num_timesteps, device=device)
    t_axis = torch.linspace(0, sampler.duration, signal_length)

    # Create subplots: one row per amplitude class
    fig, axes = plt.subplots(
        num_classes,
        num_timesteps,
        figsize=(3 * num_timesteps, 2 * num_classes),
        sharex=True,
        sharey=True,
    )

    # Handle case where there's only one class
    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, amplitude in enumerate(unique_amplitudes):
        # Get all samples with this amplitude class
        mask = labels.squeeze() == amplitude
        class_z = z[mask]
        num_class_samples = class_z.shape[0]

        for tidx, t in enumerate(ts):
            tt = t.expand(num_class_samples, 1, 1)  # shape (num_class_samples, 1, 1)
            xt = (
                path.sample_conditional_path(class_z, tt).detach().cpu()
            )  # (num_class_samples, 1, signal_length)
            xt = xt.squeeze(1)  # (num_class_samples, signal_length)

            ax = axes[class_idx, tidx]

            # Plot all samples in this class
            for sidx in range(num_class_samples):
                ax.plot(t_axis.cpu(), xt[sidx], alpha=0.7)

            if class_idx == 0:
                ax.set_title(f"t={float(t):.2f}", fontsize=10)
            if tidx == 0:
                ax.set_ylabel(f"Amp={amplitude:.1f}", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Gaussian conditional path for sine waves (grouped by amplitude class)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


def visualize_wave_path():
    num_samples = 5
    num_timesteps = 5

    sampler = WaveSampler()
    signal_length = sampler.sample_rate * sampler.duration
    path = GaussianConditionalProbabilityPath(
        p_data=sampler,
        p_simple_shape=[3, signal_length],
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)

    z, labels = path.p_data.sample(
        num_samples
    )  # z shape (num_samples, 3, signal_len), labels shape (num_samples, 1)

    # Get unique amplitude classes and group samples by class
    unique_amplitudes = torch.unique(labels).cpu()
    num_classes = len(unique_amplitudes)

    ts = torch.linspace(0, 1, num_timesteps, device=device)
    t_axis = torch.linspace(0, sampler.duration, signal_length)
    wave_types = ["Sine", "Sawtooth", "Square"]

    # Create subplots: 3 channels × num_classes rows, num_timesteps columns
    fig, axes = plt.subplots(
        3 * num_classes,
        num_timesteps,
        figsize=(3 * num_timesteps, 2 * 3 * num_classes),
        sharex=True,
        sharey="row",
    )

    axes = axes.reshape(3 * num_classes, num_timesteps)

    for class_idx, amplitude in enumerate(unique_amplitudes):
        # Get all samples with this amplitude class
        mask = labels.squeeze() == amplitude
        class_z = z[mask]
        num_class_samples = class_z.shape[0]

        for channel in range(3):
            row_idx = class_idx * 3 + channel

            for tidx, t in enumerate(ts):
                tt = t.expand(
                    num_class_samples, 1, 1
                )  # shape (num_class_samples, 1, 1)
                xt = (
                    path.sample_conditional_path(class_z, tt).detach().cpu()
                )  # (num_class_samples, 3, signal_length)

                ax = axes[row_idx, tidx]

                # Plot all samples for this channel
                for sidx in range(num_class_samples):
                    ax.plot(t_axis.cpu(), xt[sidx, channel], alpha=0.7)

                # Titles and labels
                if row_idx == 0:
                    ax.set_title(f"t={float(t):.2f}", fontsize=10)
                if tidx == 0:
                    ax.set_ylabel(
                        f"{wave_types[channel]}\nAmp={amplitude:.1f}",
                        fontsize=9,
                        fontweight="bold",
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Gaussian conditional path for waves by channel and amplitude", fontsize=14
    )
    plt.tight_layout()
    plt.show()


def visualize_generated_mnist_samples(
    path: ConditionalProbabilityPath, model: ConditionalVectorField, null_class: int = 0
):
    samples_per_class = 10
    num_timesteps = 100
    guidance_scales = [1.0, 3.0, 5.0]

    # Graph
    fig, axes = plt.subplots(
        1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10)
    )

    for idx, w in enumerate(guidance_scales):
        # Setup ode and simulator
        ode = CFGVectorFieldODE(model, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        y = (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
            .repeat_interleave(samples_per_class)
            .to(device)
        )
        num_samples = y.shape[0]
        x0, _ = path.p_simple.sample(num_samples)  # (num_samples, 1, 32, 32)

        # Simulate
        ts = (
            torch.linspace(0, 1, num_timesteps)
            .view(1, -1, 1, 1, 1)
            .expand(num_samples, -1, 1, 1, 1)
            .to(device)
        )
        x1 = simulator.simulate(x0, ts, y=y, null_class=null_class)

        # Plot
        grid = make_grid(
            x1, nrow=samples_per_class, normalize=True, value_range=(-1, 1)
        )
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
    plt.show()


def visualize_generated_sine_waves(
    model: ConditionalVectorField,
    samples_per_amplitude: int = 3,
    num_timesteps: int = 100,
    guidance_scales=(1.0,),
    null_class: int = 0,
):
    """
    Generate sine waves per amplitude class via the trained model, and plot them grouped by amplitude.

    Args:
        - model: trained conditional vector field model
        - samples_per_amplitude: number of samples to generate per amplitude class
        - num_timesteps: number of time steps for ODE simulation
        - guidance_scales: tuple of guidance scale values to test
        - null_class: the null class label for classifier-free guidance
    """
    model.eval()

    # Infer signal length and amplitude classes from the sine sampler
    sampler = SineWaveSampler()
    signal_length = sampler.sample_rate * sampler.duration
    t_axis = torch.linspace(0, sampler.duration, signal_length, device=device)
    amplitudes = sampler.amplitudes  # e.g., [1, 2, 3]
    num_classes = len(amplitudes)

    # Use actual amplitude values: [1, 1, 1, 2, 2, 2, 3, 3, 3, ...]
    amplitude_values = torch.tensor(
        amplitudes, dtype=torch.long, device=device
    ).repeat_interleave(samples_per_amplitude)
    num_samples = amplitude_values.shape[0]

    # Initial noise and time discretization
    x0 = torch.randn(num_samples, 1, signal_length, device=device)  # (bs, 1, L)
    ts = torch.linspace(0, 1, num_timesteps, device=device)  # (nts,)
    ts = ts.view(1, -1, 1, 1).expand(num_samples, -1, 1, 1)  # (bs, nts, 1, 1)

    # Create subplots: rows = amplitude classes, cols = guidance scales
    fig, axes = plt.subplots(
        num_classes,
        len(guidance_scales),
        figsize=(8 * len(guidance_scales), 4 * num_classes),
        squeeze=False,
    )

    with torch.no_grad():
        for col_idx, w in enumerate(guidance_scales):
            ode = CFGVectorFieldODE(model, guidance_scale=float(w))
            simulator = EulerSimulator(ode)
            x1 = simulator.simulate(
                x0.clone(), ts, y=amplitude_values, null_class=null_class
            )  # (bs, 1, L)

            # Plot samples grouped by amplitude class
            for class_idx, amplitude in enumerate(amplitudes):
                ax = axes[class_idx, col_idx]

                # Get indices for this amplitude class
                start_idx = class_idx * samples_per_amplitude
                end_idx = start_idx + samples_per_amplitude

                # Plot all samples in this amplitude class
                for sample_idx in range(start_idx, end_idx):
                    ax.plot(
                        t_axis.cpu(),
                        x1[sample_idx, 0].detach().cpu(),
                        alpha=0.7,
                        label=f"Sample {sample_idx - start_idx + 1}",
                    )

                # Set titles and labels
                if class_idx == 0:
                    ax.set_title(
                        f"Guidance: w={float(w):.1f}", fontsize=14, fontweight="bold"
                    )
                if col_idx == 0:
                    ax.set_ylabel(
                        f"Amplitude: {amplitude}", fontsize=12, fontweight="bold"
                    )

                ax.set_xlabel("Time")
                ax.legend(loc="upper right", fontsize="small")
                ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Generated Sine Waves by Amplitude Class", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def visualize_generated_waves(
    model: ConditionalVectorField,
    name: str,
    samples_per_amplitude: int = 1,
    num_timesteps: int = 100,
    guidance_scales=(1.0,),
    null_class: int = 0,
):
    """
    Generate waves per amplitude class via the trained model, showing each channel separately.
    Always saves the plot to plots/signals/{name}.png

    Args:
        - model: trained conditional vector field model
        - name: required name for the saved plot (without extension)
        - samples_per_amplitude: number of samples to generate per amplitude class
        - num_timesteps: number of time steps for ODE simulation
        - guidance_scales: tuple of guidance scale values to test
        - null_class: the null class label for classifier-free guidance
    """

    model.eval()

    # Infer signal length and amplitude classes from the wave sampler
    sampler = WaveSampler()
    signal_length = sampler.sample_rate * sampler.duration
    t_axis = torch.linspace(0, sampler.duration, signal_length, device=device)
    amplitudes = sampler.amplitudes  # e.g., [1, 2, 3]
    num_classes = len(amplitudes)
    wave_types = ["Sine", "Sawtooth", "Square"]

    # Use actual amplitude values: [1, 1, 2, 2, 3, 3, ...]
    amplitude_values = torch.tensor(
        amplitudes, dtype=torch.long, device=device
    ).repeat_interleave(samples_per_amplitude)
    num_samples = amplitude_values.shape[0]

    # Initial noise and time discretization
    x0 = torch.randn(num_samples, 3, signal_length, device=device)  # (bs, 3, L)
    ts = torch.linspace(0, 1, num_timesteps, device=device)  # (nts,)
    ts = ts.view(1, -1, 1, 1).expand(num_samples, -1, 1, 1)  # (bs, nts, 1, 1)

    # Create subplots: rows = 3 channels × num_classes, cols = guidance scales
    fig, axes = plt.subplots(
        3 * num_classes,
        len(guidance_scales),
        figsize=(8 * len(guidance_scales), 2 * 3 * num_classes),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    with torch.no_grad():
        for col_idx, w in enumerate(guidance_scales):
            ode = CFGVectorFieldODE(model, guidance_scale=float(w))
            simulator = EulerSimulator(ode)
            x1 = simulator.simulate(
                x0.clone(), ts, y=amplitude_values, null_class=null_class
            )  # (bs, 3, L)

            # Plot samples grouped by amplitude class and channel
            for class_idx, amplitude in enumerate(amplitudes):
                for channel in range(3):
                    row_idx = class_idx * 3 + channel
                    ax = axes[row_idx, col_idx]

                    # Get indices for this amplitude class
                    start_idx = class_idx * samples_per_amplitude
                    end_idx = start_idx + samples_per_amplitude

                    # Plot all samples in this amplitude class for this channel
                    for sample_idx in range(start_idx, end_idx):
                        ax.plot(
                            t_axis.cpu(),
                            x1[sample_idx, channel].detach().cpu(),
                            alpha=0.7,
                            label=f"Sample {sample_idx - start_idx + 1}",
                        )

                    # Set titles and labels
                    if row_idx == 0:
                        ax.set_title(
                            f"Guidance: w={float(w):.1f}",
                            fontsize=14,
                            fontweight="bold",
                        )
                    if col_idx == 0:
                        ax.set_ylabel(
                            f"{wave_types[channel]}\nAmp={amplitude}",
                            fontsize=10,
                            fontweight="bold",
                        )

                    ax.set_xlabel("Time")
                    if samples_per_amplitude > 1:
                        ax.legend(loc="upper right", fontsize="small")
                    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Generated Waves by Channel and Amplitude Class", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = "plots/signals"
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_generated_data_samples(
    model: ConditionalVectorField,
    name: str,
    samples_per_activity: int = 1,
    num_timesteps: int = 100,
    guidance_scales=(1.0,),
    null_class: int = 6,
):
    """
    Generate WISDM activity samples via the trained model and visualize them by activity class.
    All channels are shown in a single plot per activity class.

    Args:
        - model: trained conditional vector field model
        - samples_per_activity: number of samples to generate per activity class
        - num_timesteps: number of time steps for ODE simulation
        - guidance_scales: tuple of guidance scale values to test
        - null_class: the null class label for classifier-free guidance
    """
    model.eval()

    # Activity mapping
    activity_names = {
        0: "Walking",
        1: "Jogging",
        2: "Upstairs",
        3: "Downstairs",
        4: "Sitting",
        5: "Standing",
    }

    # Infer signal length from the WISDM sampler
    sampler = DataSampler(dataset="wisdm", window_time=6.0)

    # Get one sample to infer shape
    sample_data, _ = sampler.sample(1)
    signal_length = sample_data.shape[2]  # (batch, channels, signal_length)
    num_channels = sample_data.shape[1]

    # Get the number of activity classes
    num_classes = len(activity_names)
    t_axis = torch.linspace(0, 1, signal_length, device=device)

    # Create activity labels: [0, 0, 1, 1, 2, 2, ...]
    activity_labels = torch.tensor(
        list(range(num_classes)), dtype=torch.long, device=device
    ).repeat_interleave(samples_per_activity)
    num_samples = activity_labels.shape[0]

    # Initial noise and time discretization
    x0 = torch.randn(
        num_samples, num_channels, signal_length, device=device
    )  # (bs, 3, L)
    ts = torch.linspace(0, 1, num_timesteps, device=device)  # (nts,)
    ts = ts.view(1, -1, 1, 1).expand(num_samples, -1, 1, 1)  # (bs, nts, 1, 1)

    # Create subplots: rows = num_classes, cols = guidance scales
    fig, axes = plt.subplots(
        num_classes,
        len(guidance_scales),
        figsize=(8 * len(guidance_scales), 3 * num_classes),
        squeeze=False,
    )

    channel_names = ["X-axis", "Y-axis", "Z-axis"]
    channel_colors = ["#FF6B6B", "#CDC54E", "#45B7D1"]

    with torch.no_grad():
        for col_idx, w in enumerate(guidance_scales):
            ode = CFGVectorFieldODE(model, guidance_scale=float(w))
            simulator = EulerSimulator(ode)
            x1 = simulator.simulate(
                x0.clone(), ts, y=activity_labels, null_class=null_class
            )  # (bs, 3, L)

            # Plot samples grouped by activity class
            for activity_idx in range(num_classes):
                ax = axes[activity_idx, col_idx]

                # Get indices for this activity class
                start_idx = activity_idx * samples_per_activity
                end_idx = start_idx + samples_per_activity

                # Plot all channels for all samples in this activity class
                for sample_idx in range(start_idx, end_idx):
                    for channel in range(num_channels):
                        ax.plot(
                            t_axis.cpu(),
                            x1[sample_idx, channel].detach().cpu(),
                            alpha=0.7,
                            color=channel_colors[channel],
                            label=channel_names[channel]
                            if sample_idx == start_idx
                            else "",
                        )

                # Set titles and labels
                if activity_idx == 0:
                    ax.set_title(
                        f"Guidance: w={float(w):.1f}",
                        fontsize=14,
                        fontweight="bold",
                    )
                if col_idx == 0:
                    activity_name = activity_names.get(
                        activity_idx, f"Activity {activity_idx}"
                    )
                    ax.set_ylabel(activity_name, fontsize=12, fontweight="bold")

                ax.set_xlabel("Normalized Time")
                if col_idx == len(guidance_scales) - 1:
                    ax.legend(loc="upper right", fontsize="small")
                ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Generated WISDM Activity Samples (All Channels)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = "plots/signals"
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    plt.show()
