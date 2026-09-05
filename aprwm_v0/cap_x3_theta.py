"""CAP-X3 active scene vector: 10-D (drop frozen Coulomb slots)."""

from __future__ import annotations

import numpy as np

from .capx_arm3_plant import (
    DAMPING_SCALE_RANGE,
    INERTIA_SCALE_RANGE,
    MASS_SCALE_RANGE,
    NOMINAL_DAMPING,
    PAYLOAD_RANGE,
    THETA_DIM,
    SceneTheta,
)

# mass(3), inertia(3), damping(3), payload(1). frictionloss [9:12] stays 0.
ACTIVE_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12], dtype=np.int64)
D_ACTIVE = int(ACTIVE_IDX.size)
ACTIVE_NAMES = (
    "m1",
    "m2",
    "m3",
    "I1",
    "I2",
    "I3",
    "b1",
    "b2",
    "b3",
    "payload",
)


def theta_to_active(theta: SceneTheta | np.ndarray) -> np.ndarray:
    vec = theta.as_vector() if isinstance(theta, SceneTheta) else np.asarray(theta, dtype=np.float64)
    return vec[ACTIVE_IDX].copy()


def active_to_theta(active: np.ndarray) -> SceneTheta:
    a = np.asarray(active, dtype=np.float64).reshape(D_ACTIVE)
    vec = np.zeros(THETA_DIM, dtype=np.float64)
    vec[ACTIVE_IDX] = a
    return SceneTheta.from_vector(vec)


def active_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [MASS_SCALE_RANGE[0]] * 3
        + [INERTIA_SCALE_RANGE[0]] * 3
        + [float(NOMINAL_DAMPING[0] * DAMPING_SCALE_RANGE[0])] * 3
        + [PAYLOAD_RANGE[0]],
        dtype=np.float64,
    )
    hi = np.array(
        [MASS_SCALE_RANGE[1]] * 3
        + [INERTIA_SCALE_RANGE[1]] * 3
        + [float(NOMINAL_DAMPING[0] * DAMPING_SCALE_RANGE[1])] * 3
        + [PAYLOAD_RANGE[1]],
        dtype=np.float64,
    )
    return lo, hi


def nominal_active() -> np.ndarray:
    return np.concatenate(
        [np.ones(6, dtype=np.float64), np.asarray(NOMINAL_DAMPING, dtype=np.float64), np.array([0.5], dtype=np.float64)]
    )


def relative_e_theta(hat: np.ndarray, true: np.ndarray) -> float:
    t = np.asarray(true, dtype=np.float64)
    h = np.asarray(hat, dtype=np.float64)
    return float(np.linalg.norm(h - t) / (np.linalg.norm(t) + 1.0e-8))
