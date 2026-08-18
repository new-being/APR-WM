"""VIS-X visual host: same hinge dynamics as simx_hinge.v1 + camera.

Does not replace HOST_PLANT_ID for SIM-X. Physics invariants match
``simx_plant`` (I declared, b0, drive, no contact). Only observation
channel assets are added.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .simx_plant import (
    SIMX_ARMATURE,
    SIMX_DAMPING,
    SIMX_FRICTIONLOSS,
    assert_host_structure,
    learner_nominal_params,
)

VIS_HOST_PLANT_ID = "simx_hinge_vis.v1"
VIS_CAMERA_NAME = "fixed_cam"
VIS_PANEL_GEOM = "arm_panel"
# World-frame hinge pivot (arm body origin).
VIS_PIVOT_XYZ = (0.0, 0.0, 0.5)

SIMX_HINGE_VIS_MJCF = """
<mujoco model="simx_hinge_vis">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
  </visual>
  <default>
    <joint limited="false" damping="{damping}" frictionloss="{frictionloss}"
           armature="{armature}"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.7 0.75 0.8" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="panel" rgba="0.85 0.25 0.15 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid" pos="0 0 0"/>
    <camera name="fixed_cam" pos="0.6 -0.9 0.85" xyaxes="1 0.4 0 0 0.3 1" fovy="45"/>
    <body name="arm" pos="0 0 0.5">
      <inertial pos="0.15 0 0" mass="2.0" diaginertia="0.05 0.08 0.05"/>
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom name="arm_panel" type="box" size="0.2 0.02 0.3" pos="0.2 0 0"
            density="200" material="panel"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def simx_hinge_vis_xml(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> str:
    damping = SIMX_DAMPING if true_damping is None else float(true_damping)
    return SIMX_HINGE_VIS_MJCF.format(
        timestep=float(timestep),
        damping=damping,
        frictionloss=SIMX_FRICTIONLOSS,
        armature=SIMX_ARMATURE,
    )


def simx_hinge_vis_xml_sha256(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> str:
    return hashlib.sha256(
        simx_hinge_vis_xml(timestep, true_damping=true_damping).encode("utf-8")
    ).hexdigest()


def load_simx_hinge_vis(
    timestep: float,
    *,
    true_damping: float | None = None,
) -> tuple[Any, Any]:
    from .mujoco_physics import import_mujoco_with_renderer

    mujoco = import_mujoco_with_renderer()
    model = mujoco.MjModel.from_xml_string(
        simx_hinge_vis_xml(timestep, true_damping=true_damping)
    )
    data = mujoco.MjData(model)
    return model, data


def assert_vis_plant(
    model: Any,
    *,
    true_damping: float | None = None,
) -> dict[str, Any]:
    """Same kinematic/drive invariants as SIM-X host + camera/panel present.

    ``true_damping`` may differ from learner ``b0`` for VIS-X2 mismatch cells.
    """

    inv = assert_host_structure(model)
    expected = SIMX_DAMPING if true_damping is None else float(true_damping)
    if abs(inv["dof_damping"][0] - expected) > 1.0e-12:
        raise RuntimeError(
            f"VIS host XML damping must equal declared plant damping={expected}"
        )
    from .mujoco_physics import import_mujoco_with_renderer

    mujoco = import_mujoco_with_renderer()
    cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, VIS_CAMERA_NAME))
    panel_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, VIS_PANEL_GEOM))
    if cam_id < 0:
        raise RuntimeError(f"missing camera {VIS_CAMERA_NAME}")
    if panel_id < 0:
        raise RuntimeError(f"missing geom {VIS_PANEL_GEOM}")
    inv = dict(inv)
    inv["host_plant_id"] = VIS_HOST_PLANT_ID
    inv["camera_name"] = VIS_CAMERA_NAME
    inv["camera_id"] = cam_id
    inv["panel_geom"] = VIS_PANEL_GEOM
    inv["panel_geom_id"] = panel_id
    inv["dynamics_sibling_of"] = "simx_hinge.v1"
    inv["plant_true_damping"] = expected
    inv["learner_nominal"] = {
        "damping": learner_nominal_params().damping,
        "frictionloss": learner_nominal_params().frictionloss,
        "armature": learner_nominal_params().armature,
    }
    return inv
