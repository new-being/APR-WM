"""Map network output to (x0, eps). Three frozen prediction types. No clip."""

from __future__ import annotations

import torch

from .diffusion_schedule import CosineSchedule

PTYPES = ("epsilon", "v_prediction", "sample")


def alpha_sigma(sched: CosineSchedule, t: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (t.shape[0],) + (1,) * (like.ndim - 1)
    return sched.sqrt_abar[t].view(*shape), sched.sqrt_one_minus[t].view(*shape)


def diffusion_target(clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, sched: CosineSchedule, ptype: str) -> torch.Tensor:
    a, s = alpha_sigma(sched, t, clean)
    if ptype == "epsilon":
        return noise
    if ptype == "v_prediction":
        return a * noise - s * clean
    if ptype == "sample":
        return clean
    raise ValueError(ptype)


def decode_x0_eps(model_out: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, sched: CosineSchedule, ptype: str) -> tuple[torch.Tensor, torch.Tensor]:
    a, s = alpha_sigma(sched, t, xt)
    if ptype == "epsilon":
        eps = model_out
        x0 = (xt - s * eps) / torch.clamp(a, min=1e-8)
        return x0, eps
    if ptype == "v_prediction":
        v = model_out
        x0 = a * xt - s * v
        eps = s * xt + a * v
        return x0, eps
    if ptype == "sample":
        x0 = model_out
        eps = (xt - a * x0) / torch.clamp(s, min=1e-8)
        return x0, eps
    raise ValueError(ptype)
