from typing import List, Tuple

import torch
import torch.nn as nn

from .base import Sampleable


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
        self._gen = None

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
            - samples: (num_samples, 3, signal_length)
            - labels: (num_samples, 1) with class index repeated
        """
        num_data_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)
        t = torch.linspace(
            0, self.duration, self.sample_rate * self.duration, device=self.dummy.device
        )

        if self._gen is not None and self._gen.device != self.dummy.device:
            self._gen = torch.Generator(device=self.dummy.device)
            self._gen.manual_seed(self._seed)

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
        frequencies = torch.clamp(frequencies, min=1e-6)

        amps = amplitudes_tensor[class_indices - 1].unsqueeze(1)
        freqs = frequencies.unsqueeze(1)

        sine_waves = amps * torch.sin(2 * torch.pi * freqs * t + self.phase)
        sawtooth_waves = amps * (2 * (freqs * t - torch.floor(0.5 + freqs * t)))
        square_waves = amps * torch.sign(
            torch.sin(2 * torch.pi * freqs * t + self.phase)
        )

        waves = torch.stack([sine_waves, sawtooth_waves, square_waves], dim=1)
        labels = class_indices.unsqueeze(1)

        return waves, labels
