from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """ 
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass
    
class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...
        
    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None
    
class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
    
class SineWaveSampler(nn.Module, Sampleable):
    """
    Sampleable sine wave generator with random amplitude, frequency, and phase
    """
    def __init__(self, amplitude_max: float = 1.0, frequency_max: float = 1.0, phase_max: float = 2 * torch.pi, sample_rate: float = 100.0, duration: float = 1.0):
        super().__init__()
        self.amplitude_max = amplitude_max
        self.frequency_max = frequency_max
        self.phase_max = phase_max
        self.sample_rate = sample_rate
        self.duration = duration
        
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (num_samples, signal_length)
            - labels: shape (num_samples, 3) containing [amplitude, cos(phase), sin(phase)]
        """
        t = torch.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Generate random parameters for each sample
        amplitudes = torch.rand(num_samples) * self.amplitude_max
        frequencies = torch.rand(num_samples) * self.frequency_max
        phases = torch.rand(num_samples) * self.phase_max
        
        waves = []
        for i in range(num_samples):
            wave = amplitudes[i] * torch.sin(2 * torch.pi * frequencies[i] * t + phases[i])
            waves.append(wave)
        
        samples = torch.stack(waves)
        
        # Create labels: [amplitude_norm, cos(phase), sin(phase)]
        labels = torch.stack([
            amplitudes / self.amplitude_max,
            torch.cos(phases),
            torch.sin(phases)
        ], dim=1)
        
        return samples, labels
    
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