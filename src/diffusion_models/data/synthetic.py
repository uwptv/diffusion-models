from typing import List, Tuple

import torch
import torch.nn as nn

from .base import Sampleable


class SineWaveSampler(nn.Module, Sampleable):
    """
    Sampleable sine wave generator with stochastic frequency, fixed phase and amplitude as classes
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
        self.dummy = nn.Buffer(
            torch.zeros(1)
        )  # Will automatically be moved when self.to(...) is called...

    def sample(
        self, num_samples: int, mean: float = 1.0, std: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
            - mean: mean of the normal distribution for frequencies
            - std: standard deviation of the normal distribution for frequencies
        Returns:
            - samples: shape (num_samples, channels = 1, signal_length = sample_rate * duration)
            - labels: shape (num_samples, 1) containing amplitude as class
        """
        num_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(
            self.amplitudes, device=self.dummy.device
        )  # Convert to tensor
        t = torch.linspace(
            0, self.duration, self.sample_rate * self.duration, device=self.dummy.device
        )  # (signal_length,)
        class_indices = torch.randint(
            0, num_classes, (num_samples,), device=self.dummy.device
        )  # (num_samples,)

        # Generate frequencies from normal distribution and ensure they're positive
        frequencies = (
            torch.randn(num_samples, device=self.dummy.device) * std + mean
        )  # (num_samples,)
        frequencies = torch.clamp(
            frequencies, min=1e-6
        )  # Ensure all frequencies are > 0

        # Vectorized sine wave generation
        # frequencies: (num_samples,) -> (num_samples, 1) for broadcasting
        # t: (signal_length,)
        # Result: (num_samples, signal_length)
        waves = amplitudes_tensor[class_indices].unsqueeze(1) * torch.sin(
            2 * torch.pi * frequencies.unsqueeze(1) * t + self.phase
        )
        waves = waves.unsqueeze(
            1
        )  # reshape to (num_samples, 1, signal_length) for backbone
        labels = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)

        return waves, labels


class WaveSampler(nn.Module, Sampleable):
    """
    Sampleable wave generator with stochastic frequency and amplitude as classes.
    Generates 3 channels: sine waves, sawtooth waves, and square waves.
    """

    def __init__(
        self,
        amplitudes: List[int] = [1, 2, 3],
        phase: float = 0.0,
        sample_rate: int = 100,
        duration: int = 1,
        seed: int | None = 42,
    ):
        super().__init__()
        self.amplitudes = amplitudes
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.dummy = nn.Buffer(torch.zeros(1))
        self._seed = seed
        self._gen = None
        if self._gen is not None and self._gen.device != self.dummy.device:
            self._gen = torch.Generator(device=self.dummy.device)
            self._gen.manual_seed(self._seed)

    def sample(
        self, num_samples: int, mean: float = 1.0, std: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
            - mean: mean of the normal distribution for frequencies
            - std: standard deviation of the normal distribution for frequencies
        Returns:
            - samples: (num_samples, channels=3, signal_length)
            - labels: (num_samples, 1) containing amplitude class
        """
        num_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)
        t = torch.linspace(
            0, self.duration, self.sample_rate * self.duration, device=self.dummy.device
        )
        # Ensure generator is on the same device as dummy
        if self._gen is not None and self._gen.device != self.dummy.device:
            self._gen = torch.Generator(device=self.dummy.device)
            self._gen.manual_seed(self._seed)
        class_indices = torch.randint(
            0,
            num_classes,
            (num_samples,),
            device=self.dummy.device,
            generator=self._gen,
        )

        # Generate frequencies from normal distribution and ensure they're positive
        frequencies = (
            torch.randn(num_samples, device=self.dummy.device, generator=self._gen)
            * std
            + mean
        )
        frequencies = torch.clamp(frequencies, min=1e-6)

        # Get amplitudes for each sample
        amps = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)
        freqs = frequencies.unsqueeze(1)  # (num_samples, 1)

        # Channel 0: Sine waves
        sine_waves = amps * torch.sin(
            2 * torch.pi * freqs * t + self.phase
        )  # (num_samples, signal_length)

        # Channel 1: Sawtooth waves
        sawtooth_waves = amps * (
            2 * (freqs * t - torch.floor(0.5 + freqs * t))
        )  # (num_samples, signal_length)

        # Channel 2: Square waves
        square_waves = amps * torch.sign(
            torch.sin(2 * torch.pi * freqs * t + self.phase)
        )  # (num_samples, signal_length)

        # Stack channels: (num_samples, 3, signal_length)
        waves = torch.stack([sine_waves, sawtooth_waves, square_waves], dim=1)

        labels = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)

        return waves, labels
