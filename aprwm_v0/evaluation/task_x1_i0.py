"""TASK-X1-I0: diffusion sampler closure. Oracle-epsilon tests. No clip, no retrain, no Stage B."""

from __future__ import annotations

import numpy as np
import torch

from ..policy.deterministic_ddim import ddim_step, ddim_time_grid
from ..policy.diffusion_schedule import INFER_STEPS, TRAIN_STEPS, CosineSchedule


def pred_x0_from_eps(x: torch.Tensor, eps: torch.Tensor, t: int, sched: CosineSchedule) -> torch.Tensor:
    return (x - sched.sqrt_one_minus[t] * eps) / torch.clamp(sched.sqrt_abar[t], min=1e-8)


@torch.no_grad()
def c0_oracle_reverse(x0: torch.Tensor, eps: torch.Tensor, sched: CosineSchedule, t_start: int = TRAIN_STEPS - 1) -> torch.Tensor:
    t0 = torch.full((x0.shape[0],), int(t_start), device=x0.device, dtype=torch.long)
    x = sched.q_sample(x0, t0, eps)
    ts = ddim_time_grid(sched.timesteps, INFER_STEPS)
    for i, t in enumerate(ts):
        t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
        x = ddim_step(x, eps, int(t), t_prev, sched, eta=0.0)
    return x


@torch.no_grad()
def c1_oracle_x0(x0: torch.Tensor, eps: torch.Tensor, t: int, sched: CosineSchedule) -> torch.Tensor:
    tt = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    xt = sched.q_sample(x0, tt, eps)
    return pred_x0_from_eps(xt, eps, int(t), sched)


@torch.no_grad()
def c2_model_x0(model, x0: torch.Tensor, state: torch.Tensor, task_id: torch.Tensor, t: int, sched: CosineSchedule, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tt = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    xt = sched.q_sample(x0, tt, eps)
    eps_hat = model(noisy_action=xt, diffusion_t=tt, state=state, task_id=task_id)
    x0_hat = pred_x0_from_eps(xt, eps_hat, int(t), sched)
    return x0_hat, eps_hat
