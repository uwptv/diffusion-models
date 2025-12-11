from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
from torch import nn
from utility import model_size_b, MiB

from probability_paths import GaussianConditionalProbabilityPath
from differential_equations import ConditionalVectorField

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()

class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.sample_conditioning_variable(batch_size)
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) < self.eta
        y[mask] = 10
        
        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1).to(z.device) # (batch_size, 1, 1, 1)
        x = self.path.sample_conditional_path(z, t) # (batch_size, c, h, w)

        # Step 4: Regress and output loss
        u_t_theta = self.model(x, t, y)
        u_t = self.path.conditional_vector_field(x, z, t)
        return torch.mean((u_t - u_t_theta) ** 2)