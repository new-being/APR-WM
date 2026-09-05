"""Shadow relative belief state (BEL0) — does not modify formal chain estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RelativeBeliefState:
    cumulative_surprise: float = 0.0


def registration_surprise(q_rel: float, eps: float = 1e-6) -> float:
    q = float(np.clip(q_rel, eps, 1.0))
    return -np.log(q)


def update_belief(
    state: RelativeBeliefState,
    q_rel: float,
    eps: float = 1e-6,
) -> RelativeBeliefState:
    s = registration_surprise(q_rel, eps)
    return RelativeBeliefState(cumulative_surprise=state.cumulative_surprise + s)


def mean_surprise(cumulative_surprise: float, t: int) -> float:
    return float(cumulative_surprise / max(int(t), 1))


def recent_hazard(q_history: list[float], window: int = 4, eps: float = 1e-6) -> float:
    if not q_history:
        return 0.0
    x = q_history[-window:]
    return float(max(registration_surprise(q, eps) for q in x))
