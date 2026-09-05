"""TASK-X1-I1 reconstruction audit in x0 space. Selection metric, not train loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..evaluation.ac0_offline_metrics import chunk_diversity
from ..policy.diffusion_parameterization import decode_x0_eps
from ..policy.diffusion_schedule import CosineSchedule

AUDIT_TS = (10, 50, 99)
G0_L1_T10 = 0.15
G0_L1_T50 = 0.30
G1_L1_T99 = 2.0
G2_DPRED_MULT = 10.0


@torch.no_grad()
def reconstruct_x0(
    model,
    x0: torch.Tensor,
    state: torch.Tensor,
    task_id: torch.Tensor,
    t: int,
    sched: CosineSchedule,
    eps: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    tt = torch.full((x0.shape[0],), int(t), device=x0.device, dtype=torch.long)
    xt = sched.q_sample(x0, tt, eps)
    out = model(noisy_action=xt, diffusion_t=tt, state=state, task_id=task_id)
    x0_hat, _ = decode_x0_eps(out, xt, tt, sched, prediction_type)
    return x0_hat


def rec_metrics(x0_hat: torch.Tensor, x0: torch.Tensor, norm) -> dict[str, float]:
    l1 = float((x0_hat - x0).abs().mean())
    raw_hat = norm.denorm_act(x0_hat.detach().cpu().numpy())
    raw = norm.denorm_act(x0.detach().cpu().numpy())
    return {
        "l1_norm": l1,
        "max_abs_norm": float((x0_hat - x0).abs().max()),
        "l1_raw": float(np.abs(raw_hat - raw).mean()),
        "d_pred": float(chunk_diversity(raw_hat)),
        "d_demo": float(chunk_diversity(raw)),
    }


def rec_gates(per_t: dict[str, dict[str, float]]) -> dict[str, Any]:
    g0 = bool(per_t["10"]["l1_norm"] <= G0_L1_T10 and per_t["50"]["l1_norm"] <= G0_L1_T50)
    g1 = bool(per_t["99"]["l1_norm"] <= G1_L1_T99)
    g2 = bool(per_t["99"]["d_pred"] <= G2_DPRED_MULT * max(per_t["99"]["d_demo"], 1e-12))
    return {
        "G0_mid_noise": g0,
        "G1_t99_l1": g1,
        "G2_t99_diversity": g2,
        "pass": bool(g0 and g1 and g2),
        "l1_norm": {t: per_t[t]["l1_norm"] for t in ("10", "50", "99")},
        "d_pred": {t: per_t[t]["d_pred"] for t in ("10", "50", "99")},
        "d_demo": {t: per_t[t]["d_demo"] for t in ("10", "50", "99")},
        "per_t": per_t,
    }
