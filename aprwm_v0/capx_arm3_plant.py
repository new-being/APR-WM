"""CAP-X host plant: 3-DoF serial arm ``capx_arm3.v1``.

Shared kinematic structure; scenes vary only low-dimensional ``theta_e``.
No contact. Drive via ``qfrc_applied`` only. Does not unlock R10-C0.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

HOST_PLANT_ID = "capx_arm3.v1"
N_DOF = 3
THETA_DIM = 13

# Nominal link passive / inertia scales (learner structure ψ).
NOMINAL_DAMPING = np.array([0.05, 0.05, 0.05], dtype=np.float64)
NOMINAL_FRICTIONLOSS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
NOMINAL_MASSES = np.array([1.0, 0.8, 0.55], dtype=np.float64)
NOMINAL_DIAGINERTIA = np.array(
    [
        [0.010, 0.010, 0.002],
        [0.008, 0.008, 0.0015],
        [0.004, 0.004, 0.0008],
    ],
    dtype=np.float64,
)
NOMINAL_PAYLOAD = 0.0
# CAP-X0 freezes Coulomb at 0: MuJoCo frictionloss timing ≠ -μ sign(v) enough
# to break NRMSE<1e-4. Slots remain in θ (zeros) for X1+ convention lock.
MU_MAX = 0.0
MASS_SCALE_RANGE = (0.7, 1.3)
INERTIA_SCALE_RANGE = (0.7, 1.3)
DAMPING_SCALE_RANGE = (0.5, 1.5)
PAYLOAD_RANGE = (0.0, 1.0)

CAPX_ARM3_MJCF = """
<mujoco model="capx_arm3">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <joint limited="false" damping="0.05" frictionloss="0" armature="0"/>
    <geom contype="0" conaffinity="0" rgba="0.55 0.65 0.80 1"/>
  </default>
  <worldbody>
    <body name="link1" pos="0 0 0.15">
      <inertial pos="0.12 0 0" mass="1.0" diaginertia="0.010 0.010 0.002"/>
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.24 0 0" size="0.035"/>
      <body name="link2" pos="0.24 0 0">
        <inertial pos="0.10 0 0" mass="0.8" diaginertia="0.008 0.008 0.0015"/>
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.20 0 0" size="0.030"/>
        <body name="link3" pos="0.20 0 0">
          <inertial pos="0.08 0 0" mass="0.55" diaginertia="0.004 0.004 0.0008"/>
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom name="g3" type="capsule" fromto="0 0 0 0.16 0 0" size="0.025"/>
          <body name="payload" pos="0.16 0 0">
            <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>
            <geom name="g_payload" type="sphere" size="0.03" rgba="0.85 0.45 0.25 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip()

LINK_BODY_NAMES = ("link1", "link2", "link3")
PAYLOAD_BODY_NAME = "payload"


@dataclass(frozen=True)
class SceneTheta:
    """Low-dimensional scene parameters (exact CAP-X0 layout)."""

    mass_scale: np.ndarray  # (3,)
    inertia_scale: np.ndarray  # (3,)
    damping: np.ndarray  # (3,)
    frictionloss: np.ndarray  # (3,)
    payload_mass: float

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.mass_scale, dtype=np.float64).reshape(3),
                np.asarray(self.inertia_scale, dtype=np.float64).reshape(3),
                np.asarray(self.damping, dtype=np.float64).reshape(3),
                np.asarray(self.frictionloss, dtype=np.float64).reshape(3),
                np.asarray([self.payload_mass], dtype=np.float64),
            ]
        )

    @staticmethod
    def from_vector(vec: np.ndarray) -> "SceneTheta":
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size != THETA_DIM:
            raise ValueError(f"theta dim {v.size} != {THETA_DIM}")
        return SceneTheta(
            mass_scale=v[0:3].copy(),
            inertia_scale=v[3:6].copy(),
            damping=v[6:9].copy(),
            frictionloss=v[9:12].copy(),
            payload_mass=float(v[12]),
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.as_vector().tobytes()).hexdigest()


def capx_arm3_xml(timestep: float = 0.002) -> str:
    return CAPX_ARM3_MJCF.format(timestep=float(timestep))


def capx_arm3_xml_sha256(timestep: float = 0.002) -> str:
    return hashlib.sha256(capx_arm3_xml(timestep).encode("utf-8")).hexdigest()


def sample_scene_theta(rng: np.random.Generator) -> SceneTheta:
    lo_m, hi_m = MASS_SCALE_RANGE
    lo_i, hi_i = INERTIA_SCALE_RANGE
    lo_b, hi_b = DAMPING_SCALE_RANGE
    return SceneTheta(
        mass_scale=rng.uniform(lo_m, hi_m, size=3),
        inertia_scale=rng.uniform(lo_i, hi_i, size=3),
        damping=NOMINAL_DAMPING * rng.uniform(lo_b, hi_b, size=3),
        frictionloss=rng.uniform(0.0, MU_MAX, size=3),
        payload_mass=float(rng.uniform(*PAYLOAD_RANGE)),
    )


def load_capx_arm3(timestep: float = 0.002) -> tuple[Any, Any]:
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()
    model = mujoco.MjModel.from_xml_string(capx_arm3_xml(timestep))
    data = mujoco.MjData(model)
    return model, data


def _body_id(model: Any, name: str) -> int:
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()
    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if bid < 0:
        raise RuntimeError(f"missing body {name}")
    return bid


def apply_scene_theta(model: Any, data: Any, theta: SceneTheta) -> None:
    """Write scene parameters into the MuJoCo model and refresh constants."""

    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()
    for i, name in enumerate(LINK_BODY_NAMES):
        bid = _body_id(model, name)
        model.body_mass[bid] = float(NOMINAL_MASSES[i] * theta.mass_scale[i])
        model.body_inertia[bid] = NOMINAL_DIAGINERTIA[i] * float(theta.inertia_scale[i])
        model.dof_damping[i] = float(theta.damping[i])
        model.dof_frictionloss[i] = float(theta.frictionloss[i])
    pay = _body_id(model, PAYLOAD_BODY_NAME)
    # Keep a tiny floor so MuJoCo inertia stays well-conditioned.
    model.body_mass[pay] = max(float(theta.payload_mass), 1.0e-6)
    model.body_inertia[pay] = np.array([1.0e-8, 1.0e-8, 1.0e-8], dtype=np.float64)
    mujoco.mj_setConst(model, data)


def plant_inventory(model: Any) -> dict[str, Any]:
    return {
        "host_plant_id": HOST_PLANT_ID,
        "nv": int(model.nv),
        "nq": int(model.nq),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "theta_dim": THETA_DIM,
        "geom_contype": [int(model.geom_contype[i]) for i in range(int(model.ngeom))],
        "geom_conaffinity": [
            int(model.geom_conaffinity[i]) for i in range(int(model.ngeom))
        ],
        "xml_sha256": capx_arm3_xml_sha256(float(model.opt.timestep)),
        "nominal_damping": NOMINAL_DAMPING.tolist(),
        "mu_max": MU_MAX,
    }


def assert_capx_arm3(model: Any) -> dict[str, Any]:
    inv = plant_inventory(model)
    if inv["nv"] != N_DOF or inv["nq"] != N_DOF or inv["njnt"] != N_DOF:
        raise RuntimeError(f"capx_arm3.v1 must be 3-DoF, got {inv}")
    if inv["nu"] != 0:
        raise RuntimeError("capx_arm3.v1 forbids actuators; drive qfrc_applied")
    if any(inv["geom_contype"]) or any(inv["geom_conaffinity"]):
        raise RuntimeError("capx_arm3.v1 geoms must disable contact")
    return inv


def per_dof_passive_force(qvel: np.ndarray, theta: SceneTheta) -> np.ndarray:
    """Learner-visible viscous + Coulomb passive model from θ (not qfrc_passive)."""

    v = np.asarray(qvel, dtype=np.float64).reshape(N_DOF)
    damp = -np.asarray(theta.damping, dtype=np.float64) * v
    fr = -np.asarray(theta.frictionloss, dtype=np.float64) * np.sign(v)
    fr = np.where(np.abs(v) < 1.0e-12, 0.0, fr)
    return damp + fr


def oracle_force_residual(
    model: Any,
    data: Any,
    theta: SceneTheta,
    *,
    qvel_pre: np.ndarray,
) -> np.ndarray:
    """Generalized-force residual with oracle θ passive model."""

    from .mujoco_force import (
        get_bias_force,
        get_constraint_force,
        get_mass_matrix,
        known_control_force,
    )

    mass = get_mass_matrix(model, data)
    qacc = np.asarray(data.qacc, dtype=np.float64)
    bias = get_bias_force(data)
    known = known_control_force(data)
    constraint = get_constraint_force(data)
    passive = per_dof_passive_force(qvel_pre, theta)
    return mass @ qacc + bias - known - constraint - passive


def scene_theta_to_dict(theta: SceneTheta) -> dict[str, Any]:
    return {
        "mass_scale": theta.mass_scale.tolist(),
        "inertia_scale": theta.inertia_scale.tolist(),
        "damping": theta.damping.tolist(),
        "frictionloss": theta.frictionloss.tolist(),
        "payload_mass": float(theta.payload_mass),
        "vector": theta.as_vector().tolist(),
        "fingerprint": theta.fingerprint(),
    }
