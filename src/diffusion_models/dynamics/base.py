from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, xt_dim)
            - t: time, shape (bs, 1, xt_dim)
        Returns:
            - drift_coefficient: shape (bs, c, xt_dim)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, xt_dim)
            - t: time, shape (bs, 1, xt_dim)
        Returns:
            - drift_coefficient: shape (bs, c, xt_dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, xt_dim)
            - t: time, shape (bs, 1, xt_dim)
        Returns:
            - diffusion_coefficient: shape (bs, c, xt_dim)
        """
        pass


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


class CFGVectorFieldODE(ODE):
    def __init__(
        self,
        net: ConditionalVectorField,
        null_class: int,
        guidance_scale: float = 1.0,
    ):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, x_dim)
        - t: (bs, 1, x_dim)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.full_like(y, 0)
        unguided_vector_field = self.net(x, t, unguided_y)
        return (
            1 - self.guidance_scale
        ) * unguided_vector_field + self.guidance_scale * guided_vector_field
