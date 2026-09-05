"""Flattened MLP epsilon denoiser. No U-Net, no Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class DiffusionChunkDenoiser(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_tasks: int = 3,
        horizon: int = 8,
        action_dim: int = 14,
        task_emb: int = 16,
        cond_dim: int = 128,
        time_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.task_emb = nn.Embedding(num_tasks, task_emb)
        self.cond = nn.Sequential(nn.Linear(state_dim + task_emb, cond_dim), nn.GELU())
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        in_dim = horizon * action_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        diffusion_t: torch.Tensor,
        state: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        task = self.task_emb(task_id.long().view(-1))
        cond = self.cond(torch.cat([state, task], dim=-1))
        time = self.time_emb(diffusion_t.view(-1))
        noisy_flat = noisy_action.reshape(noisy_action.shape[0], -1)
        x = torch.cat([noisy_flat, cond, time], dim=-1)
        eps = self.net(x)
        return eps.view(-1, self.horizon, self.action_dim)
