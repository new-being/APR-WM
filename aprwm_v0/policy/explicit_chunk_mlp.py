"""Deterministic chunk MLP. One forward → (B, Ha, da)."""

from __future__ import annotations

import torch
from torch import nn


class ChunkMLP(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 14, horizon: int = 8, hidden: int = 256):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.view(x.shape[0], self.horizon, self.action_dim)


class ExplicitChunkPolicy(nn.Module):
    fills_by_repeat = False
    fills_by_rollout = False
    ac0_interface = True

    def __init__(
        self,
        state_dim: int,
        process_dim: int,
        *,
        use_process: bool,
        n_tasks: int = 3,
        task_emb: int = 16,
        action_dim: int = 14,
        horizon: int = 8,
        hidden: int = 256,
    ):
        super().__init__()
        self.use_process = bool(use_process)
        self.embed = nn.Embedding(n_tasks, task_emb)
        din = state_dim + task_emb + (process_dim if self.use_process else 0)
        self.mlp = ChunkMLP(din, action_dim=action_dim, horizon=horizon, hidden=hidden)
        self.state_dim = state_dim
        self.process_dim = process_dim

    def forward(self, state: torch.Tensor, task_id: torch.Tensor, process: torch.Tensor) -> torch.Tensor:
        e = self.embed(task_id.long().view(-1))
        parts = [state, e]
        if self.use_process:
            parts.append(process)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
