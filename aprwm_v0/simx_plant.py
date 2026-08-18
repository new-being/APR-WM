"""SIM-X host plant: 1-DoF vertical revolute. Shared by SIM-X0 and SIM-X1.

X0 must close accounting on this articulation. X1 may change declared
parameters on the same XML/host; it must not swap in a new kinematic
tree and still cite X0.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .mujoco_force import NominalPassiveParams

HOST_PLANT_ID = "simx_hinge.v1"

# Explicit, learner-known. X1 may mismatch these; X0 must match them.
SIMX_DAMPING = 0.10
SIMX_FRICTIONLOSS = 0.0
SIMX_ARMATURE = 0.0

SIMX_HINGE_MJCF = """
<mujoco model="simx_hinge">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <joint limited="false" damping="{damping}" frictionloss="{frictionloss}"
           armature="{armature}"/>
    <geom contype="0" conaffinity="0" rgba="0.6 0.7 0.85 1"/>
  </default>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <inertial pos="0.15 0 0" mass="2.0" diaginertia="0.05 0.08 0.05"/>
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom name="arm_panel" type="box" size="0.2 0.02 0.3" pos="0.2 0 0"
            density="200"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def simx_hinge_xml(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> str:
    """Build host XML. ``true_damping`` is plant physics only (X1).

    Learner nominal is never taken from this argument — see
    ``learner_nominal_params``.
    """

    damping = SIMX_DAMPING if true_damping is None else float(true_damping)
    return SIMX_HINGE_MJCF.format(
        timestep=float(timestep),
        damping=damping,
        frictionloss=SIMX_FRICTIONLOSS,
        armature=SIMX_ARMATURE,
    )


def simx_hinge_xml_sha256(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> str:
    return hashlib.sha256(
        simx_hinge_xml(timestep, true_damping=true_damping).encode("utf-8")
    ).hexdigest()


def learner_nominal_params() -> NominalPassiveParams:
    """Frozen learner accounting. Never follows ``true_damping``."""

    return NominalPassiveParams(
        damping=SIMX_DAMPING,
        frictionloss=SIMX_FRICTIONLOSS,
        armature=SIMX_ARMATURE,
    )


def nominal_passive_params() -> NominalPassiveParams:
    """Alias for X0: plant and learner share b0."""

    return learner_nominal_params()


def load_simx_hinge(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> tuple[Any, Any]:
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()
    model = mujoco.MjModel.from_xml_string(
        simx_hinge_xml(timestep, true_damping=true_damping)
    )
    data = mujoco.MjData(model)
    return model, data


def plant_inventory(model: Any) -> dict[str, Any]:
    n_geom = int(model.ngeom)
    contypes = [int(model.geom_contype[i]) for i in range(n_geom)]
    conaffinities = [int(model.geom_conaffinity[i]) for i in range(n_geom)]
    return {
        "host_plant_id": HOST_PLANT_ID,
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "njnt": int(model.njnt),
        "ngeom": n_geom,
        "joint_limited": [int(model.jnt_limited[i]) for i in range(int(model.njnt))],
        "dof_damping": [float(model.dof_damping[i]) for i in range(int(model.nv))],
        "dof_frictionloss": [
            float(model.dof_frictionloss[i]) for i in range(int(model.nv))
        ],
        "dof_armature": [float(model.dof_armature[i]) for i in range(int(model.nv))],
        "geom_contype": contypes,
        "geom_conaffinity": conaffinities,
        "n_actuator": int(model.nu),
    }


def assert_host_structure(model: Any) -> dict[str, Any]:
    """Kinematic / drive invariants shared by X0 and X1."""

    inv = plant_inventory(model)
    if inv["nv"] != 1 or inv["nq"] != 1 or inv["njnt"] != 1:
        raise RuntimeError(f"SIM-X host must be 1-DoF revolute, got {inv}")
    if inv["nu"] != 0:
        raise RuntimeError("SIM-X host forbids actuators; drive qfrc_applied only")
    if any(inv["joint_limited"]):
        raise RuntimeError("SIM-X host joint must be unlimited")
    if any(inv["geom_contype"]) or any(inv["geom_conaffinity"]):
        raise RuntimeError("SIM-X host geoms must have contype=conaffinity=0")
    if abs(inv["dof_frictionloss"][0] - SIMX_FRICTIONLOSS) > 1.0e-12:
        raise RuntimeError("XML frictionloss must stay at host nominal 0")
    if abs(inv["dof_armature"][0] - SIMX_ARMATURE) > 1.0e-12:
        raise RuntimeError("XML armature must stay at host nominal 0")
    return inv


def assert_x0_plant(model: Any) -> dict[str, Any]:
    inv = assert_host_structure(model)
    if abs(inv["dof_damping"][0] - SIMX_DAMPING) > 1.0e-12:
        raise RuntimeError("X0: XML damping must match learner NominalPassiveParams")
    return inv


def assert_x1_plant(model: Any, *, true_damping: float) -> dict[str, Any]:
    inv = assert_host_structure(model)
    if abs(inv["dof_damping"][0] - float(true_damping)) > 1.0e-12:
        raise RuntimeError(
            f"X1: plant dof_damping must equal true_damping={true_damping}"
        )
    return inv
