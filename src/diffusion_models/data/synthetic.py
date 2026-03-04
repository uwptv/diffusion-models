from typing import List, Tuple

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from diffusion_models.data.base import Sampleable


class SineWaveSampler(nn.Module, Sampleable):
    """
    Sampleable sine wave generator with stochastic frequency, fixed phase and amplitude as classes.
    Class 0 is reserved for null/unconditional. Data classes: 1, 2, 3, ...
    """

    def __init__(
        self,
        amplitudes: List[int] = [1, 2, 3],
        phase: float = 0.0,
        sample_rate: int = 100,
        duration: int = int(2 * torch.pi),
    ):
        super().__init__()
        self.amplitudes = amplitudes
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_classes = len(amplitudes) + 1  # +1 for null class at index 0
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(
        self,
        num_samples: int,
        mean: float = 1.0,
        std: float = 0.5,
        class_idx: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: desired number of samples
            - mean: mean of normal distribution for frequencies
            - std: standard deviation of normal distribution for frequencies
            - class_idx: if specified, sample only from this class. If None, sample from all classes.
        Returns:
            - samples: shape (num_samples, 1, signal_length)
            - labels: shape (num_samples, 1) with class index repeated
        """
        num_data_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)
        t = torch.linspace(
            0, self.duration, self.sample_rate * self.duration, device=self.dummy.device
        )

        # If class_idx specified, use it for all samples; otherwise sample randomly
        if class_idx is not None:
            assert 1 <= class_idx <= num_data_classes, (
                f"class_idx must be in [1, {num_data_classes}]"
            )
            class_indices = torch.full(
                (num_samples,), class_idx, device=self.dummy.device, dtype=torch.long
            )
        else:
            class_indices = torch.randint(
                1, num_data_classes + 1, (num_samples,), device=self.dummy.device
            )

        frequencies = torch.randn(num_samples, device=self.dummy.device) * std + mean
        frequencies = torch.clamp(frequencies, min=1e-6)

        waves = amplitudes_tensor[class_indices - 1].unsqueeze(1) * torch.sin(
            2 * torch.pi * frequencies.unsqueeze(1) * t + self.phase
        )
        waves = waves.unsqueeze(1)
        labels = class_indices.unsqueeze(1)

        return waves, labels


class WaveSampler(nn.Module, Sampleable):
    """
    Sampleable wave generator with stochastic frequency and amplitude as classes.
    Generates 3 channels: sine waves, sawtooth waves, and square waves.
    Class 0 is reserved for null/unconditional. Data classes: 1, 2, 3, ...
    """

    def __init__(
        self,
        amplitudes: List[int] = [1, 2, 3],
        phase: float = 0.0,
        sample_rate: int = 128,
        duration: int = 1,
        seed: int | None = 42,
    ):
        super().__init__()
        self.amplitudes = amplitudes
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_classes = len(amplitudes) + 1  # +1 for null class at index 0
        self.dummy = nn.Buffer(torch.zeros(1))
        self._seed = seed
        self._gen = torch.Generator(device=self.dummy.device)
        if self._seed is not None:
            self._gen.manual_seed(self._seed)
        self.mean = torch.Tensor(
            [[[0.0850], [0.0448], [0.1261]]]
        )  # precomputed mean of dataset for all the channels, shape (1, 3, 1)
        self.std = torch.Tensor(
            [[[1.5518], [1.2797], [2.1694]]]
        )  # precomputed std of dataset for all the channels, shape (1, 3, 1)

    def reset_generator(self) -> None:
        """Reset the generator to its initial state for reproducibility."""
        self._gen = torch.Generator(device=self.dummy.device)
        if self._seed is not None:
            self._gen.manual_seed(self._seed)

    def _get_mean_std(self, waves: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute mean and std across all samples and time steps for each channel
        mean = waves.mean(dim=(0, 2), keepdim=True)
        std = waves.std(dim=(0, 2), keepdim=True) + 1e-6
        return mean, std

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the precomputed mean and std tensors."""
        return self.mean, self.std

    def normalize(self, waves: torch.Tensor) -> torch.Tensor:
        return (waves - self.mean) / self.std

    def denormalize(self, waves: torch.Tensor) -> torch.Tensor:
        return waves * self.std + self.mean

    def sample(
        self,
        num_samples: int,
        mean: float = 4.0,
        std: float = 2.0,
        class_idx: int | None = None,
        subset: str | None = None,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: desired number of samples
            - mean: mean of normal distribution for frequencies
            - std: standard deviation of normal distribution for frequencies
            - class_idx: if specified, sample only from this class. If None, sample from all classes.
            - subset: if specified, sample from this subset of data (e.g., "train", "val", "test"), not used in this synthetic sampler but included for compatibility with DataSampler interface
            - normalize: whether to normalize the samples using mean and std of the dataset
        Returns:
            - samples: (num_samples, 3, signal_length)
            - labels: (num_samples, 1) with class index repeated
        """
        num_data_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)
        t = torch.linspace(
            0, self.duration, self.sample_rate * self.duration, device=self.dummy.device
        )

        if self._gen.device != self.dummy.device:
            self._gen = torch.Generator(device=self.dummy.device)
            if self._seed is not None:
                self._gen.manual_seed(self._seed)

        # If class_idx specified, use it for all samples; otherwise sample randomly
        if class_idx is not None:
            assert 0 <= class_idx <= num_data_classes - 1, (
                f"class_idx must be in [0, {num_data_classes - 1}]"
            )
            class_indices = torch.full(
                (num_samples,),
                class_idx + 1,
                device=self.dummy.device,
                dtype=torch.long,
            )
        else:
            class_indices = torch.randint(
                1,
                num_data_classes + 1,
                (num_samples,),
                device=self.dummy.device,
                generator=self._gen,
            )

        frequencies = (
            torch.randn(num_samples, device=self.dummy.device, generator=self._gen)
            * std
            + mean
        )
        frequencies = torch.clamp(frequencies, min=1.0)

        amps = amplitudes_tensor[class_indices - 1].unsqueeze(1)
        freqs = frequencies.unsqueeze(1)

        sine_waves = amps * torch.sin(2 * torch.pi * freqs * t + self.phase)
        sawtooth_waves = amps * (2 * (freqs * t - torch.floor(0.5 + freqs * t)))
        square_waves = amps * torch.sign(
            torch.sin(2 * torch.pi * freqs * t + self.phase)
        )
        # Add noise to each wave type
        for waves in [sine_waves, sawtooth_waves, square_waves]:
            noise = 0.3 * torch.randn(
                waves.shape, device=self.dummy.device, generator=self._gen
            )
            waves += noise

        waves = torch.stack([sine_waves, sawtooth_waves, square_waves], dim=1)
        labels = class_indices.unsqueeze(1)

        if normalize:
            waves = self.normalize(waves)

        return waves, labels


if __name__ == "__main__":
    sampler = WaveSampler()
    samples, labels = sampler.sample(num_samples=1000)

    plt.figure(figsize=(10, 5))
    for j in range(1):
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(samples[j, i].cpu().detach().numpy())
            plt.title(f"Wave {i + 1}")
        plt.tight_layout()
        plt.show()
