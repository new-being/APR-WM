"""VIS-EXT0 Door Mode-A plant with RGB-D (robosuite Door + frozen Panda).

Panda is visual clutter only: frozen qpos, no contact drive. Hinge is the sole
actuated DoF. Does not unlock R10-C0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mujoco_force import NominalPassiveParams
from .mujoco_physics import import_mujoco, import_mujoco_with_renderer, prepare_offscreen_gl
from .r1_rs import (
    RS0_NOMINAL_DAMPING,
    _find_hinge_dof,
    _joint_id_for_dof,
    _mujoco_handles,
    _set_joint_passive,
)

HOST_PLANT_ID = "robosuite_door_mode_a_vis.v1"
CAMERA_NAME = "agentview"
PANEL_GEOM_NAME = "Door_panel"
DEFAULT_IMAGE = 128
# Frozen human first-frame prompt at 128×128 agentview (offline; not GT auto-prompt).
HUMAN_PROMPT_BOX_YX = (22, 128, 0, 40)  # y0, y1, x0, x1
HUMAN_PROMPT_CLICK_YX = (72, 14)  # approximate door seed inside box
LEARNER_DAMPING = float(RS0_NOMINAL_DAMPING)
LEARNER_FRICTION = 0.0  # Mode-A learner accounts friction via include_constraint
# EXT0 freezes frictionloss=0 so the sole physics factor is damping (raise vision only).
TRUE_FRICTION = 0.0


def _boot_mujoco_for_robosuite_rgb() -> Any:
    """Full renderer MuJoCo + robosuite OSC qM shim (order matters)."""

    prepare_offscreen_gl()
    mujoco = import_mujoco_with_renderer()
    import_mujoco(register=True)
    return mujoco


class DoorVisPlant:
    """Door hinge plant with agentview RGB-D (+ GT element seg for eval only)."""

    def __init__(
        self,
        *,
        seed: int = 0,
        image_height: int = DEFAULT_IMAGE,
        image_width: int = DEFAULT_IMAGE,
        true_damping: float = LEARNER_DAMPING,
        true_friction: float = TRUE_FRICTION,
    ):
        self.mujoco = _boot_mujoco_for_robosuite_rgb()
        import robosuite as suite

        self.env = suite.make(
            "Door",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=CAMERA_NAME,
            camera_heights=int(image_height),
            camera_widths=int(image_width),
            camera_depths=True,
            camera_segmentations="element",
            control_freq=20,
            ignore_done=True,
            use_latch=False,
            seed=int(seed),
        )
        self.env.reset()
        self.model, self.data = _mujoco_handles(self.env)
        self.model.opt.viscosity = 0.0
        self.model.opt.density = 0.0
        self.hinge_dof = int(_find_hinge_dof(self.model))
        self.joint_id = int(_joint_id_for_dof(self.model, self.hinge_dof))
        self.hinge_qpos_adr = int(self.model.jnt_qposadr[self.joint_id])
        self.lo, self.hi = [float(x) for x in self.model.jnt_range[self.joint_id]]
        self.camera_id = int(
            self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        )
        self.panel_geom_id = int(
            self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, PANEL_GEOM_NAME
            )
        )
        self.hinge_body_id = int(self.model.jnt_bodyid[self.joint_id])
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.dt = float(self.model.opt.timestep)
        self.fovy = float(self.model.cam_fovy[self.camera_id])
        self._cam_pos0 = np.array(self.model.cam_pos[self.camera_id], dtype=np.float64, copy=True)
        self._cam_fovy0 = float(self.model.cam_fovy[self.camera_id])
        self.robot_qpos0 = np.array(self.data.qpos, dtype=np.float64, copy=True)
        self.robot_qpos_idx = [
            i for i in range(self.model.nq) if i != self.hinge_qpos_adr
        ]
        self.robot_dof_idx = [i for i in range(self.model.nv) if i != self.hinge_dof]
        self.set_true_passive(friction=true_friction, damping=true_damping)
        self.nominal = NominalPassiveParams(
            damping=LEARNER_DAMPING, frictionloss=LEARNER_FRICTION
        )
        self._freeze_robot()
        self.mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        self.env.close()

    def set_true_passive(self, *, friction: float, damping: float) -> None:
        _set_joint_passive(
            self.model, self.joint_id, friction=float(friction), damping=float(damping)
        )
        self.true_damping = float(damping)
        self.true_friction = float(friction)

    def reset_camera(self) -> None:
        self.model.cam_pos[self.camera_id][:] = self._cam_pos0
        self.model.cam_fovy[self.camera_id] = self._cam_fovy0
        self.fovy = self._cam_fovy0

    def apply_camera_shift(self, delta_xyz: tuple[float, float, float], *, fovy_delta: float = 0.0) -> None:
        self.model.cam_pos[self.camera_id][:] = self._cam_pos0 + np.asarray(
            delta_xyz, dtype=np.float64
        )
        self.model.cam_fovy[self.camera_id] = self._cam_fovy0 + float(fovy_delta)
        self.fovy = float(self.model.cam_fovy[self.camera_id])

    def _freeze_robot(self) -> None:
        self.data.qpos[self.robot_qpos_idx] = self.robot_qpos0[self.robot_qpos_idx]
        self.data.qvel[self.robot_dof_idx] = 0.0

    def set_state(self, q: float, velocity: float) -> None:
        self.data.qpos[self.hinge_qpos_adr] = float(q)
        self.data.qvel[self.hinge_dof] = float(velocity)
        self._freeze_robot()
        if hasattr(self.data, "ctrl") and self.model.nu:
            self.data.ctrl[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def hinge_pivot_world(self) -> np.ndarray:
        R = np.asarray(self.data.xmat[self.hinge_body_id], dtype=np.float64).reshape(3, 3)
        p = np.asarray(self.data.xpos[self.hinge_body_id], dtype=np.float64)
        jpos = np.asarray(self.model.jnt_pos[self.joint_id], dtype=np.float64)
        return p + R @ jpos

    def observe_rgbd_gt(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RGB uint8, metric depth, GT element-seg (eval only)."""

        from robosuite.utils.camera_utils import get_real_depth_map

        if hasattr(self.env, "_update_observables"):
            self.env._update_observables(force=True)
        obs = self.env._get_observations(force_update=True)
        rgb = np.asarray(obs[f"{CAMERA_NAME}_image"], dtype=np.uint8)
        depth_n = np.asarray(obs[f"{CAMERA_NAME}_depth"], dtype=np.float64)
        if depth_n.ndim == 3:
            depth_n = depth_n[..., 0]
        depth = np.asarray(get_real_depth_map(self.env.sim, depth_n), dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        seg = np.asarray(obs[f"{CAMERA_NAME}_segmentation_element"], dtype=np.int32)
        return rgb, depth, seg

    def inventory(self) -> dict[str, Any]:
        return {
            "host_plant_id": HOST_PLANT_ID,
            "camera_name": CAMERA_NAME,
            "camera_id": self.camera_id,
            "panel_geom_id": self.panel_geom_id,
            "hinge_dof": self.hinge_dof,
            "joint_id": self.joint_id,
            "q_range": [self.lo, self.hi],
            "dt": self.dt,
            "fovy": self.fovy,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "learner_damping": LEARNER_DAMPING,
            "true_damping": self.true_damping,
            "true_friction": self.true_friction,
            "human_prompt_box_yx": list(HUMAN_PROMPT_BOX_YX),
            "human_prompt_click_yx": list(HUMAN_PROMPT_CLICK_YX),
            "panda_frozen": True,
            "contact_with_door": False,
        }
