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
    Sampleable sine wave generator with stochastic frequency, fixed amplitude and phase
    """
    def __init__(self, amplitude: float = 1.0, phase: float = 0.0, sample_rate: int = 100, duration: int = int(2 * torch.pi)):
        super().__init__()
        self.amplitude = amplitude
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (num_samples, signal_length = sample_rate * duration)
            - labels: shape (num_samples, 1) containing frequency
        """
        t = torch.linspace(0, self.duration, self.sample_rate * self.duration, device=self.dummy.device) # (signal_length,)
        frequencies = torch.rand(num_samples, 1, device=self.dummy.device)  # random frequencies with shape (num_samples, 1)

        # Vectorized sine wave generation
        waves = self.amplitude * torch.sin(2 * torch.pi * frequencies * t + self.phase) # (num_samples, signal_length)

        return waves, frequencies
    
def visualize_sinewave_samples(samples: torch.Tensor, labels: torch.Tensor):
    t = torch.linspace(0, int(2 * torch.pi), samples.shape[-1])
    plt.figure(figsize=(10, 6))
    for i in range(samples.shape[0]):
        freq = labels[i].item()
        plt.plot(t.cpu(), samples[i].cpu(),
                 label=f'f={freq:.3f} Hz')
    plt.title('Sine Wave Samples')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

# sampler = SineWaveSampler()
# samples, labels = sampler.sample(10)
# visualize_sinewave_samples(samples, labels)