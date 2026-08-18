"""CAP-X2 outside-library residual family (tau_perp) and rho calibration.

Learner must never see alpha/beta/eta/rho/gamma. Does not unlock R10.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .capx_arm3_plant import N_DOF, SceneTheta, per_dof_passive_force

# Frozen sampling ranges for scene residual coefficients (oracle-only).
ALPHA_RANGE = (0.02, 0.12)
BETA_RANGE = (0.05, 0.35)
ETA_RANGE = (0.05, 0.35)

RHO_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)
RESIDUAL_FAMILY_ID = "aprwm.cap_x2.tau_perp.v1"

# Cyclic (j,k) for joint i: (0->1,2), (1->2,0), (2->0,1)
_CROSS_JK = ((1, 2), (2, 0), (0, 1))


@dataclass(frozen=True)
class SceneResidualCoeffs:
    """Oracle-only residual coefficients (not visible to learners)."""

    alpha: np.ndarray  # (3,)
    beta: np.ndarray  # (3,)
    eta: np.ndarray  # (3,)

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.alpha, dtype=np.float64).reshape(3),
                np.asarray(self.beta, dtype=np.float64).reshape(3),
                np.asarray(self.eta, dtype=np.float64).reshape(3),
            ]
        )

    @staticmethod
    def from_vector(vec: np.ndarray) -> "SceneResidualCoeffs":
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size != 9:
            raise ValueError(f"residual coeff dim {v.size} != 9")
        return SceneResidualCoeffs(alpha=v[0:3].copy(), beta=v[3:6].copy(), eta=v[6:9].copy())

    def fingerprint(self) -> str:
        return hashlib.sha256(self.as_vector().tobytes()).hexdigest()


def sample_residual_coeffs(rng: np.random.Generator) -> SceneResidualCoeffs:
    lo_a, hi_a = ALPHA_RANGE
    lo_b, hi_b = BETA_RANGE
    lo_e, hi_e = ETA_RANGE
    return SceneResidualCoeffs(
        alpha=rng.uniform(lo_a, hi_a, size=3),
        beta=rng.uniform(lo_b, hi_b, size=3),
        eta=rng.uniform(lo_e, hi_e, size=3),
    )


def g_unscaled(q: np.ndarray, qd: np.ndarray, coeffs: SceneResidualCoeffs) -> np.ndarray:
    """Unscaled residual bracket g(q, qd); shape (..., 3)."""

    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None, :]
        qd = qd[None, :]
    out = np.zeros_like(q)
    for i in range(N_DOF):
        j, k = _CROSS_JK[i]
        out[:, i] = (
            coeffs.alpha[i] * qd[:, i] * np.abs(qd[:, i])
            + coeffs.beta[i] * np.sin(2.0 * q[:, i])
            + coeffs.eta[i] * np.sin(q[:, j] - q[:, k])
        )
    return out[0] if single else out


def tau_perp(
    q: np.ndarray,
    qd: np.ndarray,
    coeffs: SceneResidualCoeffs,
    *,
    gamma: float,
) -> np.ndarray:
    if float(gamma) == 0.0:
        q = np.asarray(q, dtype=np.float64)
        return np.zeros(q.shape if q.ndim > 1 else (N_DOF,), dtype=np.float64)
    return float(gamma) * g_unscaled(q, qd, coeffs)


def tau_phy_library(
    model: Any,
    data: Any,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    u: np.ndarray,
    theta: SceneTheta,
) -> np.ndarray:
    """Library generalized force RHS: u + passive - bias (no tau_perp)."""

    from .mujoco_force import get_bias_force
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()
    data.qpos[:] = np.asarray(q, dtype=np.float64)
    data.qvel[:] = np.asarray(qd, dtype=np.float64)
    if model.nu:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    bias = get_bias_force(data)
    passive = per_dof_passive_force(qd, theta)
    return np.asarray(u, dtype=np.float64) + passive - bias


def calibrate_gamma(
    *,
    r_perp: float,
    r_phy: float,
    rho: float,
    eps: float = 1.0e-12,
) -> float:
    if float(rho) == 0.0:
        return 0.0
    return float(rho) * float(r_phy) / (float(r_perp) + eps)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def residual_coeffs_to_dict(coeffs: SceneResidualCoeffs) -> dict[str, Any]:
    return {
        "family_id": RESIDUAL_FAMILY_ID,
        "alpha": coeffs.alpha.tolist(),
        "beta": coeffs.beta.tolist(),
        "eta": coeffs.eta.tolist(),
        "vector": coeffs.as_vector().tolist(),
        "fingerprint": coeffs.fingerprint(),
    }
