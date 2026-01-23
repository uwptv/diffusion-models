from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
import torch.nn as nn

from diffusion_models.data.base import Sampleable, IsotropicGaussian
from diffusion_models.dynamics.schedules import Alpha, Beta

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, x_dim)
        Returns:
            - x: samples from p_t(x), (num_samples, c, x_dim)
        """
        num_samples = t.shape[0]

        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, x_dim)

        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, x_dim)

        return x

    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, x_dim)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, x_dim)
            - t: time (num_samples, 1, x_dim)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, x_dim)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, x_dim)
            - z: conditioning variable (num_samples, c, x_dim)
            - t: time (num_samples, 1, x_dim)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, x_dim)
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, x_dim)
            - z: conditioning variable (num_samples, c, x_dim)
            - t: time (num_samples, 1, x_dim)
        Returns:
            - conditional_score: conditional score (num_samples, c, x_dim)
        """ 
        pass
    
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, ...)
            - t: time (num_samples, 1, ...)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, ...)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, ...)
            - z: conditioning variable (num_samples, c, ...)
            - t: time (num_samples, 1, ...)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, ...)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, ...)
        beta_t = self.beta(t) # (num_samples, 1, ...)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, ...)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, ...)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, ...)
            - z: conditioning variable (num_samples, c, ...)
            - t: time (num_samples, 1, ...)
        Returns:
            - conditional_score: conditional score (num_samples, c, ...)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2