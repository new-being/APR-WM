"""Adjacent relative state chain propagation (SEQ0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .relative_rigid_registration import RelativeICPConfig, RelativeRigidResult, estimate_relative_rigid


@dataclass
class ChainState:
    p: np.ndarray
    n: np.ndarray
    t_accum: np.ndarray  # 4x4 homogeneous cumulative from frame 0


def _hom(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t4 = np.eye(4, dtype=np.float64)
    t4[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t4[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return t4


def propagate_state(
    state_prev: ChainState,
    delta_r: np.ndarray,
    delta_t: np.ndarray,
) -> ChainState:
    r = np.asarray(delta_r, dtype=np.float64).reshape(3, 3)
    t = np.asarray(delta_t, dtype=np.float64).reshape(3)
    n_new = r @ state_prev.n
    n_new = n_new / max(np.linalg.norm(n_new), 1e-12)
    p_new = r @ state_prev.p + t
    t_new = _hom(r, t) @ state_prev.t_accum
    return ChainState(p=p_new, n=n_new, t_accum=t_new)


def run_adjacent_chain(
    clouds: list[np.ndarray],
    p0_hat: np.ndarray,
    n0_hat: np.ndarray,
    object_diameter: float,
    *,
    cfg: RelativeICPConfig | None = None,
    checkpoints: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> tuple[dict[int, ChainState], list[RelativeRigidResult]]:
    """Propagate using frozen adjacent ICP; clouds[t] aligned with state at t."""
    cfg = cfg or RelativeICPConfig()
    n0 = np.asarray(n0_hat, dtype=np.float64).reshape(3)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    state = ChainState(p=np.asarray(p0_hat, dtype=np.float64).reshape(3), n=n0, t_accum=np.eye(4))
    ckpt_set = set(int(h) for h in checkpoints)
    ckpt_states: dict[int, ChainState] = {0: state}
    step_results: list[RelativeRigidResult] = []
    for t in range(1, len(clouds)):
        res = estimate_relative_rigid(clouds[t - 1], clouds[t], object_diameter, cfg)
        step_results.append(res)
        if res.valid:
            state = propagate_state(state, res.R, res.t)
        if t in ckpt_set:
            ckpt_states[t] = ChainState(
                p=state.p.copy(), n=state.n.copy(), t_accum=state.t_accum.copy(),
            )
    return ckpt_states, step_results


def run_direct_anchor(
    cloud0: np.ndarray,
    cloud_t: np.ndarray,
    p0_hat: np.ndarray,
    n0_hat: np.ndarray,
    object_diameter: float,
    *,
    cfg: RelativeICPConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, RelativeRigidResult]:
    cfg = cfg or RelativeICPConfig()
    res = estimate_relative_rigid(cloud0, cloud_t, object_diameter, cfg)
    n0 = np.asarray(n0_hat, dtype=np.float64).reshape(3)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    p0 = np.asarray(p0_hat, dtype=np.float64).reshape(3)
    if not res.valid:
        return p0.copy(), n0.copy(), res
    n_t = res.R @ n0
    n_t = n_t / max(np.linalg.norm(n_t), 1e-12)
    p_t = res.R @ p0 + res.t
    return p_t, n_t, res
