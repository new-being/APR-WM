"""Cosine noise schedule. Train K=100 frozen. No sweep."""

from __future__ import annotations

import numpy as np
import torch

TRAIN_STEPS = 100
INFER_STEPS = 10


def cosine_betas(timesteps: int = TRAIN_STEPS, s: float = 0.008) -> np.ndarray:
    steps = int(timesteps) + 1
    x = np.linspace(0.0, timesteps, steps, dtype=np.float64)
    ac = np.cos(((x / timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / np.maximum(ac[:-1], 1e-8))
    return np.clip(betas, 1e-5, 0.999)


class CosineSchedule:
    def __init__(self, timesteps: int = TRAIN_STEPS, device: str | torch.device = "cpu"):
        betas = cosine_betas(timesteps)
        alphas = 1.0 - betas
        abar = np.cumprod(alphas)
        self.timesteps = int(timesteps)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
        self.abar = torch.tensor(abar, dtype=torch.float32, device=device)
        self.sqrt_abar = torch.sqrt(self.abar)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.abar)

    def to(self, device: str | torch.device) -> "CosineSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.abar = self.abar.to(device)
        self.sqrt_abar = self.sqrt_abar.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        shape = (t.shape[0],) + (1,) * (x0.ndim - 1)
        sa = self.sqrt_abar[t].view(*shape)
        so = self.sqrt_one_minus[t].view(*shape)
        return sa * x0 + so * noise
