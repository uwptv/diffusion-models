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
    Sampleable sine wave generator with stochastic frequency, fixed phase and amplitude as classes
    """
    def __init__(self, amplitudes: List[int] = [1, 2, 3], phase: float = 0.0, sample_rate: int = 100, duration: int = int(2 * torch.pi)):
        super().__init__()
        self.amplitudes = amplitudes
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int, mean: float = 1.0, std: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
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
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)  # Convert to tensor
        t = torch.linspace(0, self.duration, self.sample_rate * self.duration, device=self.dummy.device) # (signal_length,)
        class_indices = torch.randint(0, num_classes, (num_samples,), device=self.dummy.device)  # (num_samples,)
        
        # Generate frequencies from normal distribution and ensure they're positive
        frequencies = torch.randn(num_samples, device=self.dummy.device) * std + mean  # (num_samples,)
        frequencies = torch.clamp(frequencies, min=1e-6)  # Ensure all frequencies are > 0
        
        # Vectorized sine wave generation
        # frequencies: (num_samples,) -> (num_samples, 1) for broadcasting
        # t: (signal_length,)
        # Result: (num_samples, signal_length)
        waves = amplitudes_tensor[class_indices].unsqueeze(1) * torch.sin(2 * torch.pi * frequencies.unsqueeze(1) * t + self.phase)
        waves = waves.unsqueeze(1)  # reshape to (num_samples, 1, signal_length) for backbone
        labels = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)

        return waves, labels
    
def visualize_sinewave_samples(samples: torch.Tensor, labels: torch.Tensor):
    """
    Visualize sine wave samples grouped by amplitude class.
    
    Args:
        - samples: shape (num_samples, channels = 1, signal_length)
        - labels: shape (num_samples, 1) containing amplitude values
    """
    t = torch.linspace(0, int(2 * torch.pi), samples.shape[-1])
    
    # Get unique amplitude classes
    unique_amplitudes = torch.unique(labels).cpu().numpy()
    num_classes = len(unique_amplitudes)
    
    # Create subplots, one for each amplitude class
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 4 * num_classes))
    
    # Handle case where there's only one class (axes won't be an array)
    if num_classes == 1:
        axes = [axes]
    
    for class_idx, amplitude in enumerate(unique_amplitudes):
        # Find all samples with this amplitude class
        mask = (labels.squeeze() == amplitude)
        class_samples = samples[mask]
        
        # Plot all samples in this class
        for i in range(class_samples.shape[0]):
            axes[class_idx].plot(t.cpu(), class_samples[i, 0].cpu(), 
                               alpha=0.7, label=f'Sample {i+1}')
        
        axes[class_idx].set_title(f'Amplitude: {amplitude:.1f}')
        axes[class_idx].set_xlabel('Time')
        axes[class_idx].set_ylabel('Signal Value')
        axes[class_idx].legend(loc='upper right', fontsize='small')
        axes[class_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# sampler = SineWaveSampler()
# samples, labels = sampler.sample(10)
# visualize_sinewave_samples(samples, labels)

class WaveSampler(nn.Module, Sampleable):
    '''
    Sampleable wave generator with stochastic frequency and amplitude as classes.
    Generates 3 channels: sine waves, sawtooth waves, and square waves.
    '''
    def __init__(self, amplitudes: List[int] = [1, 2, 3], phase: float = 0.0, sample_rate: int = 100, duration: int = int(2 * torch.pi)):
        super().__init__()
        self.amplitudes = amplitudes
        self.phase = phase
        self.sample_rate = sample_rate
        self.duration = duration
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(self, num_samples: int, mean: float = 1.0, std: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            - num_samples: the desired number of samples
            - mean: mean of the normal distribution for frequencies
            - std: standard deviation of the normal distribution for frequencies
        Returns:
            - samples: (num_samples, channels=3, signal_length)
            - labels: (num_samples, 1) containing amplitude class
        '''
        num_classes = len(self.amplitudes)
        amplitudes_tensor = torch.tensor(self.amplitudes, device=self.dummy.device)
        t = torch.linspace(0, self.duration, self.sample_rate * self.duration, device=self.dummy.device)
        class_indices = torch.randint(0, num_classes, (num_samples,), device=self.dummy.device)
        
        # Generate frequencies from normal distribution and ensure they're positive
        frequencies = torch.randn(num_samples, device=self.dummy.device) * std + mean
        frequencies = torch.clamp(frequencies, min=1e-6)
        
        # Get amplitudes for each sample
        amps = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)
        freqs = frequencies.unsqueeze(1)  # (num_samples, 1)
        
        # Channel 0: Sine waves
        sine_waves = amps * torch.sin(2 * torch.pi * freqs * t + self.phase)  # (num_samples, signal_length)
        
        # Channel 1: Sawtooth waves
        sawtooth_waves = amps * (2 * (freqs * t - torch.floor(0.5 + freqs * t)))  # (num_samples, signal_length)
        
        # Channel 2: Square waves
        square_waves = amps * torch.sign(torch.sin(2 * torch.pi * freqs * t + self.phase))  # (num_samples, signal_length)
        
        # Stack channels: (num_samples, 3, signal_length)
        waves = torch.stack([sine_waves, sawtooth_waves, square_waves], dim=1)
        
        labels = amplitudes_tensor[class_indices].unsqueeze(1)  # (num_samples, 1)
        
        return waves, labels

def visualize_wave_samples(samples: torch.Tensor, labels: torch.Tensor):
    t = torch.linspace(0, int(2 * torch.pi), samples.shape[-1])
    wave_types = ['Sine', 'Sawtooth', 'Square']
    
    # Get unique amplitude classes
    unique_amplitudes = torch.unique(labels).cpu().numpy()
    num_classes = len(unique_amplitudes)
    
    # Create subplots: 3 rows (one per wave type)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for wave_idx in range(3):
        for class_idx, amplitude in enumerate(unique_amplitudes):
            # Find samples with this amplitude
            mask = (labels.squeeze() == amplitude)
            class_samples = samples[mask]
            
            # Plot samples from this class
            for i in range(class_samples.shape[0]):
                axes[wave_idx].plot(t.cpu(), class_samples[i, wave_idx].cpu(), 
                                   alpha=0.6, label=f'Amp={amplitude:.1f}')
        
        axes[wave_idx].set_title(f'{wave_types[wave_idx]} Waves', fontsize=12, fontweight='bold')
        axes[wave_idx].set_xlabel('Time')
        axes[wave_idx].set_ylabel('Amplitude')
        axes[wave_idx].grid(True, alpha=0.3)
        # Remove duplicate legend entries
        handles, labels_legend = axes[wave_idx].get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        axes[wave_idx].legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()

# wave_sampler = WaveSampler()
# samples, labels = wave_sampler.sample(5)
# visualize_wave_samples(samples, labels)