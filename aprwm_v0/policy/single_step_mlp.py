"""Single-step 256×3 GELU MLP. Ha=1. No RNN/Transformer/process."""

from __future__ import annotations

import torch
from torch import nn

from ..data.history_windows import HS, PAST_A
from .task_embedding import STATE_DIM


HIDDEN = 256
ACTION_DIM = 14
TASK_EMB = 16


def input_dim(kind: str) -> int:
    if kind == "B0":
        return STATE_DIM + TASK_EMB
    if kind == "B1":
        return STATE_DIM * HS + TASK_EMB
    if kind == "B2":
        return STATE_DIM * HS + ACTION_DIM * PAST_A + TASK_EMB
    raise ValueError(kind)


class SingleStepMLP(nn.Module):
    fills_by_repeat = False
    fills_by_rollout = False
    uses_process = False
    horizon = 1

    def __init__(self, kind: str, n_tasks: int = 3, hidden: int = HIDDEN):
        super().__init__()
        self.kind = str(kind)
        self.embed = nn.Embedding(n_tasks, TASK_EMB)
        din = input_dim(self.kind)
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, ACTION_DIM),
        )

    def _feat(self, states: torch.Tensor, task_id: torch.Tensor, past_actions: torch.Tensor | None) -> torch.Tensor:
        e = self.embed(task_id.long().view(-1))
        if self.kind == "B0":
            st = states[:, -1, :]
            return torch.cat([st, e], dim=-1)
        flat_s = states.reshape(states.shape[0], -1)
        if self.kind == "B1":
            return torch.cat([flat_s, e], dim=-1)
        assert past_actions is not None
        return torch.cat([flat_s, past_actions.reshape(past_actions.shape[0], -1), e], dim=-1)

    def forward(self, states: torch.Tensor, task_id: torch.Tensor, past_actions: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self._feat(states, task_id, past_actions))
