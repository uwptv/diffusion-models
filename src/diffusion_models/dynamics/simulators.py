from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from .base import ODE, SDE


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, xt_dim)
            - t: time, shape (bs, 1, xt_dim)
            - dt: time, shape (bs, 1, xt_dim)
        Returns:
            - nxt: state at time t + dt (bs, c, xt_dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state, shape (bs, c, xt_dim)
            - ts: timesteps, shape (bs, nts, 1, xt_dim)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, xt_dim)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretization gives by ts
        Args:
            - x: initial state, shape (bs, c, xt_dim)
            - ts: timesteps, shape (bs, nts, 1, xt_dim)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, xt_dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return (
            xt
            + self.sde.drift_coefficient(xt, t, **kwargs) * h
            + self.sde.diffusion_coefficient(xt, t, **kwargs)
            * torch.sqrt(h)
            * torch.randn_like(xt)
        )
