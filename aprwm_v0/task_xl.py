"""TASK-XL causal encoder API stub.

Design freeze only: no training, no diffusion, no TASK-X1, no R10.
See REPORT/REG/TASKX/TASKXL_PREREG.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_ID = "aprwm.task_xl.causal_encoder.v0"
SUPERVISION_FIRST_INSTRUMENT = "B"  # future action as TARGET; primary test
DIFFUSION_TRAIN = False
UNLOCKS_R10 = False
UNLOCKS_TASK_X1 = False


class CausalLeakError(ValueError):
    """Encoder saw future states/actions (illegal)."""


def _as_2d(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank 1 or 2, got {arr.ndim}")
    return arr


def assert_causal_prefix(
    t: int,
    s_all: np.ndarray,
    a_all: np.ndarray | None = None,
) -> None:
    """Refuse any window that includes times > t."""
    if t < 0:
        raise CausalLeakError(f"t must be >= 0, got {t}")
    s = _as_2d(s_all, name="s")
    if s.shape[0] > t + 1:
        raise CausalLeakError(
            f"encoder saw s_{{t+1:T}}: len(s)={s.shape[0]} > t+1={t + 1}"
        )
    if a_all is None:
        return
    a = _as_2d(a_all, name="a")
    if a.shape[0] > t:
        raise CausalLeakError(
            f"encoder saw a_{{t:T}} as input: len(a)={a.shape[0]} > t={t}"
        )


def refuse_locked_output(path: str | Any) -> None:
    text = str(path).replace("\\", "/")
    if "r10_c0" in text:
        raise RuntimeError("r10_c0 is locked; TASK-XL must not write there")
    if "task_x1" in text or "taskx1" in text:
        raise RuntimeError("TASK-X1 / diffusion is locked; TASK-XL must not write there")


@dataclass(frozen=True)
class CausalHistory:
    """h_t = (s_{0:t}, a_{0:t-1})."""

    s_leq_t: np.ndarray
    a_lt: np.ndarray
    t: int

    @classmethod
    def from_episode(
        cls,
        t: int,
        s_all: np.ndarray,
        a_all: np.ndarray,
    ) -> CausalHistory:
        s = _as_2d(s_all, name="s")
        a = _as_2d(a_all, name="a")
        if t >= s.shape[0]:
            raise CausalLeakError(f"t={t} requires s_0..s_t; only {s.shape[0]} states")
        assert_causal_prefix(t, s[: t + 1], a[:t] if t > 0 else np.zeros((0, a.shape[1])))
        s_pref = s[: t + 1]
        a_pref = a[:t] if t > 0 else np.zeros((0, a.shape[1]), dtype=np.float64)
        return cls(s_leq_t=s_pref, a_lt=a_pref, t=t)


class CausalTaskProgressEncoder:
    """z_g = E_g(g); z^p_t = E_p(h_t, z_g). Deterministic stub, not a trained net."""

    def __init__(self, *, z_g_dim: int = 4, z_p_dim: int = 4) -> None:
        self.z_g_dim = int(z_g_dim)
        self.z_p_dim = int(z_p_dim)

    def encode_task(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64).reshape(-1)
        if g.size == 0:
            g = np.zeros(1, dtype=np.float64)
        z = np.zeros(self.z_g_dim, dtype=np.float64)
        take = min(self.z_g_dim, g.size)
        z[:take] = g[:take]
        nrm = np.linalg.norm(z)
        if nrm > 0:
            z = z / nrm
        return z

    def encode_progress(
        self,
        history: CausalHistory,
        z_g: np.ndarray,
        *,
        s_future: np.ndarray | None = None,
        a_future: np.ndarray | None = None,
        success_future: Any = None,
    ) -> np.ndarray:
        if s_future is not None or a_future is not None or success_future is not None:
            raise CausalLeakError(
                "future may be TARGET only; illegal as encoder input "
                "(s_{t+1:T}, future a, or future success)"
            )
        assert_causal_prefix(history.t, history.s_leq_t, history.a_lt)
        z_g = np.asarray(z_g, dtype=np.float64).reshape(-1)
        s_last = history.s_leq_t[-1]
        z = np.zeros(self.z_p_dim, dtype=np.float64)
        mix = np.concatenate([z_g, s_last[: min(8, s_last.size)]])
        take = min(self.z_p_dim, mix.size)
        z[:take] = mix[:take]
        z = z + 0.01 * float(history.t)
        return z

    def is_prefix_recoverable(
        self,
        s_all: np.ndarray,
        a_all: np.ndarray,
        g: np.ndarray,
    ) -> bool:
        s = _as_2d(s_all, name="s")
        a = _as_2d(a_all, name="a")
        z_g = self.encode_task(g)
        T = s.shape[0]
        for t in range(T):
            h_pref = CausalHistory.from_episode(t, s[: t + 1], a[:t] if t else a[:0])
            h_full_cut = CausalHistory.from_episode(t, s, a)
            z_a = self.encode_progress(h_pref, z_g)
            z_b = self.encode_progress(h_full_cut, z_g)
            if not np.allclose(z_a, z_b):
                return False
        return True


def physics_inputs_ok(keys: set[str]) -> bool:
    """F_physics must not consume task/progress latents."""
    forbidden = {"z_g", "z_p", "z^p", "p_t", "task_latent", "progress_latent"}
    return forbidden.isdisjoint(keys)


def action_attractor_inputs_ok(keys: set[str]) -> bool:
    """q(A|·) eats s^phy, z_g, z^p — not a diffusion train hook."""
    required = {"s_phy", "z_g", "z_p"}
    return required.issubset(keys) and DIFFUSION_TRAIN is False
