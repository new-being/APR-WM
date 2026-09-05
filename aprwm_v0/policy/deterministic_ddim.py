"""DDIM eta=0, N=10. Deterministic given generator."""

from __future__ import annotations

import numpy as np
import torch

from .diffusion_parameterization import decode_x0_eps
from .diffusion_schedule import INFER_STEPS, CosineSchedule


def ddim_time_grid(train_steps: int, n_infer: int = INFER_STEPS) -> np.ndarray:
    # inclusive of 0; descending
    return np.round(np.linspace(train_steps - 1, 0, n_infer)).astype(np.int64)


@torch.no_grad()
def ddim_step(
    x: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    t_prev: int,
    sched: CosineSchedule,
    eta: float = 0.0,
) -> torch.Tensor:
    if eta != 0.0:
        raise ValueError("TASK-X1 freezes eta=0")
    abar_t = sched.abar[t]
    sqrt_ab_t = sched.sqrt_abar[t]
    sqrt_om_t = sched.sqrt_one_minus[t]
    pred_x0 = (x - sqrt_om_t * eps) / torch.clamp(sqrt_ab_t, min=1e-8)
    if t_prev < 0:
        return pred_x0
    return ddim_step_x0(pred_x0, eps, t_prev, sched, eta=eta)


@torch.no_grad()
def ddim_step_x0(
    pred_x0: torch.Tensor,
    eps: torch.Tensor,
    t_prev: int,
    sched: CosineSchedule,
    eta: float = 0.0,
) -> torch.Tensor:
    if eta != 0.0:
        raise ValueError("TASK-X1 freezes eta=0")
    if t_prev < 0:
        return pred_x0
    abar_prev = sched.abar[t_prev]
    dir_xt = torch.sqrt(torch.clamp(1.0 - abar_prev, min=0.0)) * eps
    return torch.sqrt(abar_prev) * pred_x0 + dir_xt


@torch.no_grad()
def ddim_sample(
    model,
    state: torch.Tensor,
    task_id: torch.Tensor,
    sched: CosineSchedule,
    generator: torch.Generator,
    n_infer: int = INFER_STEPS,
    eta: float = 0.0,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    b = int(state.shape[0])
    h = int(model.horizon)
    d = int(model.action_dim)
    x = torch.randn(b, h, d, generator=generator).to(device=state.device, dtype=state.dtype)
    ts = ddim_time_grid(sched.timesteps, n_infer)
    for i, t in enumerate(ts):
        t_prev = int(ts[i + 1]) if i + 1 < len(ts) else -1
        t_batch = torch.full((b,), int(t), device=state.device, dtype=torch.long)
        out = model(noisy_action=x, diffusion_t=t_batch, state=state, task_id=task_id)
        x0, eps = decode_x0_eps(out, x, t_batch, sched, prediction_type)
        x = ddim_step_x0(x0, eps, t_prev, sched, eta=eta)
    return x
