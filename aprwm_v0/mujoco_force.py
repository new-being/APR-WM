"""MuJoCo generalized-force adapter for R1-MJ/RS.

Uses the post-3.11 CSR mass-matrix path and never reads truth
``qfrc_passive`` into the learner residual.  Truth passive forces may be
logged under ``raw_truth`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NominalPassiveParams:
    """Learner-side passive model; must not be read from ``data.qfrc_passive``."""

    damping: float = 0.0
    frictionloss: float = 0.0
    armature: float = 0.0


def _as_f64(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.dtype == object:
        raise TypeError("Expected numeric MuJoCo buffer")
    return np.array(array, copy=True)


def get_mass_matrix(model: Any, data: Any) -> np.ndarray:
    """Return dense joint-space inertia ``M(q)`` without using removed ``qM``."""

    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()

    nv = int(model.nv)
    dense = np.zeros((nv, nv), dtype=np.float64)
    # MuJoCo >= 3.11: mj_fullM(model, data, dst)
    # Older transitional API: mj_fullM(model, dst, qM-like) — rejected here.
    try:
        mujoco.mj_fullM(model, data, dense)
    except TypeError as exc:
        # Prefer explicit CSR densification over the deleted qM path.
        if not hasattr(data, "M"):
            raise RuntimeError(
                "MuJoCo mass matrix requires mjData.M (CSR); qM is not used"
            ) from exc
        mujoco.mju_sym2dense(
            dense,
            data.M,
            nv,
            model.M_rownnz,
            model.M_rowadr,
            model.M_colind,
        )
    return dense


def get_bias_force(data: Any) -> np.ndarray:
    return _as_f64(data.qfrc_bias)


def get_constraint_force(data: Any) -> np.ndarray:
    if hasattr(data, "qfrc_constraint"):
        return _as_f64(data.qfrc_constraint)
    return np.zeros(int(np.asarray(data.qvel).shape[0]), dtype=np.float64)


def get_actuator_force(data: Any) -> np.ndarray:
    if hasattr(data, "qfrc_actuator"):
        return _as_f64(data.qfrc_actuator)
    return np.zeros(int(np.asarray(data.qvel).shape[0]), dtype=np.float64)


def get_applied_force(data: Any) -> np.ndarray:
    if hasattr(data, "qfrc_applied"):
        return _as_f64(data.qfrc_applied)
    return np.zeros(int(np.asarray(data.qvel).shape[0]), dtype=np.float64)


def get_truth_passive_force(data: Any) -> np.ndarray:
    """Audit-only truth passive force; never feed into learner residual."""

    if hasattr(data, "qfrc_passive"):
        return _as_f64(data.qfrc_passive)
    return np.zeros(int(np.asarray(data.qvel).shape[0]), dtype=np.float64)


def nominal_passive_force(
    qvel: np.ndarray,
    params: NominalPassiveParams | np.ndarray,
    *,
    dof_index: int | None = None,
) -> np.ndarray:
    """Compute learner-visible passive force from declared parameters only.

    Prefer a full per-DOF parameter vector for multi-DOF systems.  A scalar
    ``NominalPassiveParams`` may be used for 1-DOF models, or with
    ``dof_index`` to write a single DOF into an otherwise-zero vector.
    """

    velocity = _as_f64(qvel)
    if isinstance(params, np.ndarray):
        return _as_f64(params)

    force = np.zeros_like(velocity)
    if dof_index is None:
        if velocity.size != 1:
            raise ValueError(
                "Scalar NominalPassiveParams requires 1-DOF qvel or dof_index"
            )
        indices = (0,)
    else:
        indices = (int(dof_index),)
    for index in indices:
        damping = -float(params.damping) * velocity[index]
        if float(params.frictionloss) == 0.0:
            friction = 0.0
        elif abs(velocity[index]) < 1.0e-12:
            friction = 0.0
        else:
            friction = -float(params.frictionloss) * float(np.sign(velocity[index]))
        force[index] = damping + friction
    return force


def known_control_force(data: Any) -> np.ndarray:
    return get_actuator_force(data) + get_applied_force(data)


def generalized_force_residual(
    model: Any,
    data: Any,
    *,
    nominal_passive: NominalPassiveParams | np.ndarray,
    include_constraint: bool = True,
    dof_index: int | None = None,
) -> np.ndarray:
    """Learner residual on generalized force; excludes truth ``qfrc_passive``."""

    mass = get_mass_matrix(model, data)
    qacc = _as_f64(data.qacc)
    bias = get_bias_force(data)
    known = known_control_force(data)
    constraint = get_constraint_force(data) if include_constraint else np.zeros_like(known)
    passive = nominal_passive_force(
        _as_f64(data.qvel), nominal_passive, dof_index=dof_index
    )
    return mass @ qacc + bias - known - constraint - passive


def residual_nrmse(
    residual: np.ndarray,
    reference_force: np.ndarray,
    *,
    eps: float = 1.0e-8,
) -> float:
    residual = _as_f64(residual)
    reference_force = _as_f64(reference_force)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    rms_ref = float(np.sqrt(np.mean(np.square(reference_force))))
    return rmse / (rms_ref + eps)
