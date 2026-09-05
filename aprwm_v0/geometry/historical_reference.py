"""Multi-scale discrete history retrieval (MEM-B0 B1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .periodic_reanchor import Keyframe
from .relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from .relative_state_chain import ChainState, propagate_state
from .relative_validity import symmetric_support_score

RETRIEVAL_AGES = (1, 2, 4, 8, 16, 32)


def exponential_keyframe_indices(t: int, ages: tuple[int, ...] = RETRIEVAL_AGES) -> list[int]:
    out: list[int] = []
    for a in ages:
        k = int(t) - int(a)
        if k >= 0:
            out.append(k)
    return out


@dataclass
class RetrievalPick:
    k: int
    q: float
    age: int
    R: np.ndarray
    t: np.ndarray
    n_corr: int
    rmse: float


def retrieve_reference(
    t: int,
    current_cloud: np.ndarray,
    history: dict[int, Keyframe],
    object_diameter: float,
    *,
    icp_cfg: RelativeICPConfig | None = None,
) -> RetrievalPick | None:
    candidates: list[RetrievalPick] = []
    for k in exponential_keyframe_indices(t):
        kf = history.get(k)
        if kf is None:
            continue
        delta = estimate_relative_rigid(kf.cloud, current_cloud, object_diameter, icp_cfg)
        if not delta.valid:
            continue
        q = symmetric_support_score(kf.cloud, current_cloud, delta.R, delta.t, object_diameter)
        candidates.append(RetrievalPick(
            k=int(k),
            q=float(q),
            age=int(t) - int(k),
            R=delta.R,
            t=delta.t,
            n_corr=int(delta.n_corr),
            rmse=float(delta.rmse),
        ))
    if not candidates:
        return None
    best = max(candidates, key=lambda x: (x.q, x.k))
    return best


def apply_retrieval(pick: RetrievalPick, history: dict[int, Keyframe]) -> ChainState:
    src = history[pick.k]
    return propagate_state(src.state, pick.R, pick.t)
