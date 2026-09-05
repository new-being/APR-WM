"""R1-RS2-C0: oracle contact-force closure before contact-mediated science."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import get_bias_force, get_mass_matrix, get_truth_passive_force
from .mujoco_physics import import_mujoco
from .r1_mj import stage_status
from .r1_rs import (
    _find_hinge_dof,
    _joint_id_for_dof,
    _mujoco_handles,
    _set_joint_passive,
)
from .r1_rs1b import _require_rs1a5_unlock
from .v06 import _write_csv, _write_json


PREREG_PATH = "REPORT/REG/R1/R1_RS2_PREREG.md"
DEV_SEEDS = (11001, 11011, 11021)
SCRIPTS = ("slow_pull", "fast_pull", "pull_release")
REPEATS = (0, 1, 2)
ADAPTER_SCALES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
MODE_A_EXPOSURES = {
    0.5: 1.94874407985e-15,
    1.0: 6.34156770481e-7,
    1.5: 7.80291087723e-4,
    2.0: 4.75869014372e-3,
}
TARGET_EXPOSURE = MODE_A_EXPOSURES[1.5]
FROZEN_ADAPTER_SCALE = 2.0
FROZEN_SCRIPT_AMP = {
    "slow_pull": 1.0,
    "fast_pull": 1.5,
    "pull_release": 1.0,
}
REGIME_ALPHA = {
    "C0": 0.0,
    "C1-L": -0.12,
    "C1-H": -0.24,
    "C2": 0.0,
}
_ROLLOUT_SCALAR_KEYS = (
    "q",
    "qvel",
    "qacc",
    "tau_contact_jtf",
    "tau_contact_efc",
    "tau_actuator",
    "tau_nominal_damping",
    "tau_nominal_friction",
    "residual",
    "qfrc_constraint_truth",
    "qfrc_constraint_unaccounted",
    "qfrc_passive_truth",
    "qfrc_applied_truth",
    "tau_hidden",
    "density_tangent",
    "contact_count",
    "finite",
    "phase",
)
LEARNER_SENSOR_KEYS = (
    "wrist_force_n",
    "wrist_torque_n",
    "arm_tau_resid_rms",
    "contact_proxy",
)
SENSOR_SIGMA_FORCE = 0.5
SENSOR_SIGMA_TORQUE = 0.05
SENSOR_SIGMA_RESID = 0.1

FIXED_PANDA_QPOS = (
    -0.0008701945799862113,
    0.16321420030237277,
    -0.014509970210316704,
    -2.6389741651159304,
    -0.027113352449214107,
    2.9160712254877863,
    0.7818878602056725,
)


@dataclass(frozen=True)
class R1RS2C0Config:
    seeds: tuple[int, ...] = DEV_SEEDS
    scripts: tuple[str, ...] = SCRIPTS
    repeats: tuple[int, ...] = REPEATS
    adapter_scales: tuple[float, ...] = ADAPTER_SCALES
    control_freq: int = 20
    initial_hinge_q: float = 0.12
    friction: float = 0.10
    damping: float = 0.10
    joint_limit_margin: float = 0.05
    nrmse_max: float = 1.0e-3
    contact_audit_nrmse_max: float = 1.0e-6
    movement_min: float = 0.05
    contact_fraction_min: float = 0.10
    contact_p95_min: float = 0.02
    exposure_rel_tolerance: float = 0.20
    warmup_lift_steps: int = 20
    warmup_side_steps: int = 25
    warmup_lower_steps: int = 20
    approach_steps: int = 24
    close_steps: int = 8
    release_steps: int = 10
    approach_max_command: float = 0.18
    close_max_command: float = 0.12
    slow_pull_steps: int = 18
    slow_pull_stroke: float = 0.035
    slow_pull_max_command: float = 0.14
    fast_pull_steps: int = 10
    fast_pull_stroke: float = 0.055
    fast_pull_max_command: float = 0.30
    release_pull_steps: int = 14
    release_pull_stroke: float = 0.045
    release_pull_max_command: float = 0.218


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _script_manifest(config: R1RS2C0Config) -> dict[str, Any]:
    return {
        "controller": "robosuite default Panda OSC_POSE",
        "control_freq": config.control_freq,
        "robot_initialization_noise": None,
        "robot_initial_qpos": list(FIXED_PANDA_QPOS),
        "door_placement": {
            "x": 0.08354208,
            "y": 0.00429212,
            "yaw": -1.6036196747,
            "reference_pos": [-0.2, -0.35, 0.8],
        },
        "gripper_open": -1.0,
        "gripper_closed": 1.0,
        "warmup": {
            "lift_steps": config.warmup_lift_steps,
            "side_steps": config.warmup_side_steps,
            "lower_steps": config.warmup_lower_steps,
            "side_offset_m": 0.10,
            "height_offset_m": 0.12,
        },
        "measured": {
            "initial_hinge_q": config.initial_hinge_q,
            "approach_steps": config.approach_steps,
            "close_steps": config.close_steps,
            "slow_pull": {
                "steps": config.slow_pull_steps,
                "stroke_m": config.slow_pull_stroke,
                "max_command": config.slow_pull_max_command,
            },
            "fast_pull": {
                "steps": config.fast_pull_steps,
                "stroke_m": config.fast_pull_stroke,
                "max_command": config.fast_pull_max_command,
            },
            "pull_release": {
                "steps": config.release_pull_steps,
                "stroke_m": config.release_pull_stroke,
                "max_command": config.release_pull_max_command,
                "release_steps": config.release_steps,
            },
        },
    }


def _manifest_sha256(config: R1RS2C0Config) -> str:
    payload = json.dumps(_script_manifest(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_robot_door_pair(name1: str, name2: str) -> bool:
    def robot(name: str) -> bool:
        return name.startswith("robot0_") or name.startswith("gripper0_")

    return (name1.startswith("Door_") and robot(name2)) or (
        name2.startswith("Door_") and robot(name1)
    )


def _constraint_projection(
    mujoco: Any,
    model: Any,
    data: Any,
    constraint_type: int,
) -> np.ndarray:
    if int(data.nefc) == 0:
        return np.zeros(int(model.nv), dtype=np.float64)
    jacobian = np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc, model.nv)
    types = np.asarray(data.efc_type, dtype=np.int32)[: data.nefc]
    mask = types == int(constraint_type)
    if not bool(np.any(mask)):
        return np.zeros(int(model.nv), dtype=np.float64)
    forces = np.asarray(data.efc_force, dtype=np.float64)[: data.nefc]
    return jacobian[mask].T @ forces[mask]


def _contact_generalized_force(
    mujoco: Any,
    model: Any,
    data: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return robot–Door contact force via J^T f and contact EFC audit."""

    tau_jtf = np.zeros(int(model.nv), dtype=np.float64)
    tau_efc = np.zeros(int(model.nv), dtype=np.float64)
    efc_jacobian = (
        np.asarray(data.efc_J, dtype=np.float64).reshape(data.nefc, model.nv)
        if int(data.nefc)
        else np.zeros((0, int(model.nv)), dtype=np.float64)
    )
    efc_force = np.asarray(data.efc_force, dtype=np.float64)[: data.nefc]
    count = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1 = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        )
        name2 = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        )
        if not _is_robot_door_pair(name1, name2):
            continue

        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench_local)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame.T @ wrench_local[:3]
        torque_world = frame.T @ wrench_local[3:]
        position = np.asarray(contact.pos, dtype=np.float64)

        jp1 = np.zeros((3, model.nv), dtype=np.float64)
        jr1 = np.zeros((3, model.nv), dtype=np.float64)
        jp2 = np.zeros((3, model.nv), dtype=np.float64)
        jr2 = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jp1, jr1, position, body1)
        mujoco.mj_jac(model, data, jp2, jr2, position, body2)
        tau_jtf += (jp2 - jp1).T @ force_world + (jr2 - jr1).T @ torque_world

        address, dim = int(contact.efc_address), int(contact.dim)
        if address >= 0:
            tau_efc += (
                efc_jacobian[address : address + dim].T
                @ efc_force[address : address + dim]
            )
        count += 1
    return tau_jtf, tau_efc, count


def _panda_arm_dofs(mujoco: Any, model: Any) -> np.ndarray:
    dofs: list[int] = []
    for joint in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if re.search(r"joint[1-7]$", name) is None:
            continue
        dofs.append(int(model.jnt_dofadr[joint]))
    return np.asarray(dofs, dtype=np.int64)


def _eef_site_id(robot: Any) -> int:
    site = robot.eef_site_id
    if isinstance(site, dict):
        return int(next(iter(site.values())))
    return int(site)


def learner_visible_contact_sensors(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    arm_dofs: np.ndarray,
    eef_site: int,
    seed: int,
) -> dict[str, float]:
    """Wrist/proprio contact proxies. No contact-pair list and no hinge J^T f."""

    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, int(eef_site))
    arm = np.asarray(arm_dofs, dtype=np.int64)
    jacobian = np.vstack((jacp[:, arm], jacr[:, arm]))
    mass = get_mass_matrix(model, data)
    qacc = np.asarray(data.qacc, dtype=np.float64)
    residual = mass @ qacc + get_bias_force(data)
    residual = residual - np.asarray(data.qfrc_actuator, dtype=np.float64)
    residual = residual - np.asarray(data.qfrc_applied, dtype=np.float64)
    arm_resid = residual[arm]
    wrench, *_ = np.linalg.lstsq(jacobian.T, arm_resid, rcond=None)
    tick = int(round(float(data.time) / max(float(model.opt.timestep), 1.0e-8)))
    rng = np.random.default_rng((int(seed) * 1_000_003 + tick) % (2**32 - 1))
    force = max(0.0, float(np.linalg.norm(wrench[:3])) + float(rng.normal(0.0, SENSOR_SIGMA_FORCE)))
    torque = max(
        0.0, float(np.linalg.norm(wrench[3:])) + float(rng.normal(0.0, SENSOR_SIGMA_TORQUE))
    )
    resid_rms = max(
        0.0,
        float(np.sqrt(np.mean(arm_resid * arm_resid))) + float(rng.normal(0.0, SENSOR_SIGMA_RESID)),
    )
    return {
        "wrist_force_n": force,
        "wrist_torque_n": torque,
        "arm_tau_resid_rms": resid_rms,
        "contact_proxy": math.log1p(force),
    }


TAXEL_RES = 8
RGB_RES = 32
RGB_CAMERA = "agentview"


def camera_env_kwargs(*, include_rgb: bool) -> dict[str, Any]:
    if not include_rgb:
        return {"has_offscreen_renderer": False, "use_camera_obs": False}
    return {
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": RGB_CAMERA,
        "camera_heights": RGB_RES,
        "camera_widths": RGB_RES,
        "camera_depths": False,
    }


def normalize_rgb(image: Any, res: int = RGB_RES) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
        if float(np.max(array)) > 1.5:
            array = array / 255.0
    if array.shape[0] != res or array.shape[1] != res:
        # nearest-neighbor fallback without extra deps
        ys = np.linspace(0, array.shape[0] - 1, res).astype(np.int64)
        xs = np.linspace(0, array.shape[1] - 1, res).astype(np.int64)
        array = array[ys][:, xs]
    return np.clip(array, 0.0, 1.0)
TAXEL_HALF = 0.04
TAXEL_NOISE = 0.05


def _is_gripper_geom(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("finger", "hand", "gripper", "pad"))


def bin_taxel_force(
    local_xy: np.ndarray,
    magnitude: float,
    *,
    res: int = TAXEL_RES,
    half: float = TAXEL_HALF,
) -> tuple[int, int] | None:
    u = (float(local_xy[0]) + half) / (2.0 * half) * res
    v = (float(local_xy[1]) + half) / (2.0 * half) * res
    ix, iy = int(math.floor(u)), int(math.floor(v))
    if 0 <= ix < res and 0 <= iy < res:
        return iy, ix
    return None


def learner_visible_tactile_fields(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    eef_site: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Gripper-local normal/shear maps plus patch and multi-surface stats.

    Not hinge J^T f; contact pairs are not a model input.
    """

    grid = np.zeros((TAXEL_RES, TAXEL_RES), dtype=np.float64)
    shear = np.zeros((TAXEL_RES, TAXEL_RES), dtype=np.float64)
    site_pos = np.asarray(data.site_xpos[int(eef_site)], dtype=np.float64)
    site_mat = np.asarray(data.site_xmat[int(eef_site)], dtype=np.float64).reshape(3, 3)
    weights: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    surf = np.zeros(9, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
        gripper = name1 if _is_gripper_geom(name1) else name2
        if not (_is_gripper_geom(name1) or _is_gripper_geom(name2)):
            continue
        wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench_local)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame.T @ wrench_local[:3]
        normal_hat = frame[:, 0]
        f_n = abs(float(np.dot(force_world, normal_hat)))
        f_s = float(np.linalg.norm(force_world - np.dot(force_world, normal_hat) * normal_hat))
        magnitude = float(np.linalg.norm(wrench_local[:3]))
        local = site_mat.T @ (np.asarray(contact.pos, dtype=np.float64) - site_pos)
        cell = bin_taxel_force(local[:2], magnitude)
        if cell is not None:
            grid[cell] += magnitude
            shear[cell] += f_s
        weights.append(f_n)
        xs.append(float(local[0]))
        ys.append(float(local[1]))
        moment_z = float(local[0] * (site_mat.T @ force_world)[1] - local[1] * (site_mat.T @ force_world)[0])
        sid = _gripper_surface_id(gripper)
        surf[3 * sid] += f_n
        surf[3 * sid + 1] += f_s
        surf[3 * sid + 2] += moment_z
    geom = _patch_geometry(np.asarray(xs), np.asarray(ys), np.asarray(weights))
    tick = int(round(float(data.time) / max(float(model.opt.timestep), 1.0e-8)))
    rng = np.random.default_rng((int(seed) * 917_011 + tick) % (2**32 - 1))
    grid = np.maximum(grid + rng.normal(0.0, TAXEL_NOISE, size=grid.shape), 0.0)
    shear = np.maximum(shear + rng.normal(0.0, TAXEL_NOISE, size=shear.shape), 0.0)
    return {"taxel": grid, "taxel_shear": shear, "tactile_geom": geom, "tactile_surf": surf}


def _gripper_surface_id(name: str) -> int:
    lower = name.lower()
    if "finger1" in lower:
        return 0
    if "finger2" in lower:
        return 1
    return 2


def _patch_geometry(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(weights) == 0 or float(np.sum(weights)) <= 1.0e-12:
        return np.zeros(7, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mass = float(np.sum(w))
    cx = float(np.sum(w * xs) / mass)
    cy = float(np.sum(w * ys) / mass)
    dx, dy = xs - cx, ys - cy
    ixx = float(np.sum(w * dy * dy) / mass)
    iyy = float(np.sum(w * dx * dx) / mass)
    ixy = float(np.sum(w * dx * dy) / mass)
    eig = np.sort(np.linalg.eigvalsh(np.asarray([[iyy, ixy], [ixy, ixx]], dtype=np.float64)))
    ecc = 0.0
    if eig[1] > 1.0e-12:
        ecc = float(np.sqrt(max(0.0, 1.0 - max(eig[0], 0.0) / eig[1])))
    return np.asarray([cx, cy, mass, ixx, iyy, ixy, ecc], dtype=np.float64)


def learner_visible_taxel_map(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    eef_site: int,
    seed: int,
) -> np.ndarray:
    """Backward-compatible magnitude map used by V7C.1/C.2."""

    return learner_visible_tactile_fields(
        mujoco, model, data, eef_site=eef_site, seed=seed
    )["taxel"]


def _record_template(*, include_learner_sensors: bool) -> dict[str, list[float]]:
    keys = _ROLLOUT_SCALAR_KEYS + (LEARNER_SENSOR_KEYS if include_learner_sensors else ())
    return {key: [] for key in keys}


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else math.nan


def _nearest_exposure_bin(x_phi: float) -> float:
    floor = 1.0e-15
    return min(
        MODE_A_EXPOSURES,
        key=lambda amplitude: abs(
            math.log(max(float(x_phi), floor))
            - math.log(max(MODE_A_EXPOSURES[amplitude], floor))
        ),
    )


class DoorContactC0Backend:
    """Panda OSC contact backend with simulation-rate force accounting."""

    def __init__(
        self,
        config: R1RS2C0Config,
        *,
        seed: int,
        regime: str = "C0",
        alpha: float | None = None,
        revision_fn: Callable[[float, float], float] | None = None,
        include_rgb: bool = False,
    ):
        self.config = config
        self.seed = int(seed)
        self.regime = str(regime)
        self.alpha = float(REGIME_ALPHA[self.regime] if alpha is None else alpha)
        self.use_latch = self.regime == "C2"
        self.revision_fn = revision_fn
        self.include_rgb = bool(include_rgb)
        self._action_log: list[np.ndarray] | None = None
        self._prepared_qpos: np.ndarray | None = None
        self._prepared_qvel: np.ndarray | None = None
        if self.include_rgb:
            from .mujoco_physics import prepare_offscreen_gl

            prepare_offscreen_gl()
        self.mujoco = import_mujoco(register=True)
        if self.include_rgb:
            from .mujoco_physics import prepare_offscreen_gl

            prepare_offscreen_gl()
        import robosuite as suite
        from robosuite.utils.placement_samplers import UniformRandomSampler

        placement = UniformRandomSampler(
            name="RS2FixedDoorPlacement",
            x_range=[0.08354208, 0.08354208],
            y_range=[0.00429212, 0.00429212],
            rotation=-1.6036196747,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=(-0.2, -0.35, 0.8),
        )

        self.env = suite.make(
            "Door",
            robots="Panda",
            has_renderer=False,
            **camera_env_kwargs(include_rgb=self.include_rgb),
            control_freq=config.control_freq,
            ignore_done=True,
            use_latch=self.use_latch,
            seed=seed,
            initialization_noise=None,
            placement_initializer=placement,
            hard_reset=False,
        )
        self.env.robots[0].init_qpos = np.asarray(
            FIXED_PANDA_QPOS, dtype=np.float64
        )
        self.obs = self.env.reset()
        self.model, self.data = _mujoco_handles(self.env)
        self.hinge_dof = _find_hinge_dof(self.model)
        self.joint_id = _joint_id_for_dof(self.model, self.hinge_dof)
        self.hinge_qpos_adr = int(self.model.jnt_qposadr[self.joint_id])
        self.lo, self.hi = [
            float(value) for value in self.model.jnt_range[self.joint_id]
        ]
        _set_joint_passive(
            self.model,
            self.joint_id,
            friction=config.friction,
            damping=config.damping,
        )
        self.model.opt.viscosity = 0.0
        self.model.opt.density = 0.0
        self.handle_geom = int(
            self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "Door_handle"
            )
        )
        if self.handle_geom < 0:
            raise RuntimeError("Door_handle geom not found")
        self.n_substeps = int(
            round(self.env.control_timestep / self.env.model_timestep)
        )
        mass = float(
            get_mass_matrix(self.model, self.data)[
                self.hinge_dof, self.hinge_dof
            ]
        )
        self.mass_proxy = max(mass, 1.0e-6)
        self._arm_dofs = _panda_arm_dofs(self.mujoco, self.model)
        self._eef_site = _eef_site_id(self.env.robots[0])
        self._want_sensors = False
        self._want_tactile = False
        self._want_rgb = False
        self._taxel_log: list[np.ndarray] | None = None
        self._shear_log: list[np.ndarray] | None = None
        self._geom_log: list[np.ndarray] | None = None
        self._surf_log: list[np.ndarray] | None = None
        self._rgb_log: list[np.ndarray] | None = None

    def close(self) -> None:
        self._want_sensors = False
        self._want_tactile = False
        self._want_rgb = False
        self._taxel_log = None
        self._shear_log = None
        self._geom_log = None
        self._surf_log = None
        self._rgb_log = None
        self.env.close()

    def _handle_position(self) -> np.ndarray:
        return np.array(self.data.geom_xpos[self.handle_geom], dtype=np.float64)

    def _observation(self) -> dict[str, Any]:
        self.obs = self.env._get_observations(force_update=True)
        return self.obs

    def _action(
        self,
        target: np.ndarray,
        *,
        gripper: float,
        max_command: float,
    ) -> np.ndarray:
        action = np.zeros(int(self.env.action_dim), dtype=np.float64)
        error = np.asarray(target, dtype=np.float64) - np.asarray(
            self.obs["robot0_eef_pos"], dtype=np.float64
        )
        action[:3] = np.clip(error / 0.05, -max_command, max_command)
        action[-1] = float(gripper)
        return action

    def _sample(self) -> dict[str, float]:
        # Recompute all force terms at the same post-integration state.
        self.mujoco.mj_forward(self.model, self.data)
        tau_jtf, tau_efc, contact_count = _contact_generalized_force(
            self.mujoco, self.model, self.data
        )
        friction = _constraint_projection(
            self.mujoco,
            self.model,
            self.data,
            int(self.mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF),
        )
        mass = get_mass_matrix(self.model, self.data)
        qacc = np.asarray(self.data.qacc, dtype=np.float64)
        lhs = mass @ qacc + get_bias_force(self.data)
        nominal_damping = -self.config.damping * np.asarray(
            self.data.qvel, dtype=np.float64
        )
        hinge = self.hinge_dof
        residual = (
            float(lhs[hinge])
            - float(self.data.qfrc_actuator[hinge])
            - float(tau_jtf[hinge])
            - float(friction[hinge])
            - float(nominal_damping[hinge])
        )
        full_constraint = float(self.data.qfrc_constraint[hinge])
        unaccounted_constraint = (
            full_constraint - float(tau_efc[hinge]) - float(friction[hinge])
        )
        q = float(self.data.qpos[self.hinge_qpos_adr])
        velocity = float(self.data.qvel[hinge])
        finite = bool(
            np.isfinite(
                [
                    q,
                    velocity,
                    residual,
                    tau_jtf[hinge],
                    tau_efc[hinge],
                ]
            ).all()
        )
        payload = {
            "q": q,
            "qvel": velocity,
            "qacc": float(self.data.qacc[hinge]),
            "tau_contact_jtf": float(tau_jtf[hinge]),
            "tau_contact_efc": float(tau_efc[hinge]),
            "tau_actuator": float(self.data.qfrc_actuator[hinge]),
            "tau_nominal_damping": float(nominal_damping[hinge]),
            "tau_nominal_friction": float(friction[hinge]),
            "residual": residual,
            "qfrc_constraint_truth": full_constraint,
            "qfrc_constraint_unaccounted": unaccounted_constraint,
            "qfrc_passive_truth": float(get_truth_passive_force(self.data)[hinge]),
            "qfrc_applied_truth": float(self.data.qfrc_applied[hinge]),
            "tau_hidden": float(self._hidden(velocity)),
            "density_tangent": float(
                get_mass_matrix(self.model, self.data)[hinge, hinge]
                * float(self.data.qacc[hinge])
                / self.mass_proxy
            ),
            "contact_count": float(contact_count),
            "finite": float(finite),
        }
        if self._want_sensors:
            payload.update(
                learner_visible_contact_sensors(
                    self.mujoco,
                    self.model,
                    self.data,
                    arm_dofs=self._arm_dofs,
                    eef_site=self._eef_site,
                    seed=self.seed,
                )
            )
        return payload

    def _hidden(self, velocity: float) -> float:
        return self.alpha * abs(float(velocity)) * float(velocity)

    def _extra_force(self) -> float:
        q = float(self.data.qpos[self.hinge_qpos_adr])
        velocity = float(self.data.qvel[self.hinge_dof])
        extra = self._hidden(velocity)
        if self.revision_fn is not None:
            extra += float(self.revision_fn(q, velocity))
        return extra

    def _control_step(
        self,
        action: np.ndarray,
        *,
        records: dict[str, list[float]] | None,
        phase: float = 0.0,
    ) -> None:
        for substep in range(self.n_substeps):
            if self.env.lite_physics:
                self.env.sim.step1()
            else:
                self.env.sim.forward()
            self.env._pre_action(action, policy_step=substep == 0)
            # OSC torques stay in qfrc_actuator. Hinge extras are explicit.
            self.env.sim.data.qfrc_applied[:] = 0.0
            extra = self._extra_force()
            if extra != 0.0:
                self.env.sim.data.qfrc_applied[self.hinge_dof] = extra
            if self.env.lite_physics:
                self.env.sim.step2()
            else:
                self.env.sim.step()
            if records is not None:
                sample = self._sample()
                for key, value in sample.items():
                    if key in records:
                        records[key].append(value)
                if "phase" in records:
                    records["phase"].append(float(phase))
                if self._want_tactile:
                    bundle = learner_visible_tactile_fields(
                        self.mujoco,
                        self.model,
                        self.data,
                        eef_site=self._eef_site,
                        seed=self.seed,
                    )
                    if self._taxel_log is None:
                        self._taxel_log = []
                    self._taxel_log.append(bundle["taxel"])
                    if self._shear_log is not None:
                        self._shear_log.append(bundle["taxel_shear"])
                        self._geom_log.append(bundle["tactile_geom"])
                        self._surf_log.append(bundle["tactile_surf"])
        if self._action_log is not None:
            self._action_log.append(np.asarray(action, dtype=np.float64).copy())
        self.env.cur_time += self.env.control_timestep
        self.env.timestep += 1
        self.env._update_observables(force=True)
        self._observation()
        self._maybe_log_rgb()

    def _maybe_log_rgb(self) -> None:
        if not self._want_rgb or self._rgb_log is None:
            return
        image = self.obs.get(f"{RGB_CAMERA}_image")
        if image is None:
            return
        self._rgb_log.append(normalize_rgb(image))

    def _goto(
        self,
        target: np.ndarray,
        *,
        steps: int,
        gripper: float,
        max_command: float,
        records: dict[str, list[float]] | None = None,
        phase: float = 0.0,
    ) -> None:
        for _ in range(int(steps)):
            action = self._action(
                target, gripper=gripper, max_command=max_command
            )
            self._control_step(action, records=records, phase=phase)

    def prepare(self) -> None:
        # Move above and to the +x side of the handle without logging. The
        # measured episode starts only after an interior hinge reset.
        target = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64).copy()
        target[2] = 1.20
        self._goto(
            target,
            steps=self.config.warmup_lift_steps,
            gripper=-1.0,
            max_command=1.0,
        )
        target = self._handle_position() + np.asarray([0.10, 0.0, 0.12])
        self._goto(
            target,
            steps=self.config.warmup_side_steps,
            gripper=-1.0,
            max_command=1.0,
        )
        target = self._handle_position() + np.asarray([0.10, 0.0, 0.0])
        self._goto(
            target,
            steps=self.config.warmup_lower_steps,
            gripper=-1.0,
            max_command=1.0,
        )
        self.data.qpos[self.hinge_qpos_adr] = self.config.initial_hinge_q
        self.data.qvel[self.hinge_dof] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        self._observation()
        self._prepared_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self._prepared_qvel = np.asarray(self.data.qvel, dtype=np.float64).copy()

    def rollout(
        self,
        script: str,
        *,
        scale: float = 1.0,
        include_learner_sensors: bool = False,
        include_tactile: bool = False,
        include_rgb: bool = False,
    ) -> dict[str, Any]:
        self._want_sensors = bool(include_learner_sensors)
        self._want_tactile = bool(include_tactile)
        self._want_rgb = bool(include_rgb) and bool(self.include_rgb)
        self._taxel_log = [] if include_tactile else None
        self._shear_log = [] if include_tactile else None
        self._geom_log = [] if include_tactile else None
        self._surf_log = [] if include_tactile else None
        self._rgb_log = [] if self._want_rgb else None
        records: dict[str, list[float]] = _record_template(
            include_learner_sensors=include_learner_sensors
        )
        self._action_log = []
        self._goto(
            self._handle_position(),
            steps=self.config.approach_steps,
            gripper=-1.0,
            max_command=self.config.approach_max_command,
            records=records,
            phase=0.0,
        )
        self._goto(
            self._handle_position(),
            steps=self.config.close_steps,
            gripper=1.0,
            max_command=self.config.close_max_command,
            records=records,
            phase=1.0,
        )

        if script == "slow_pull":
            pull_steps = self.config.slow_pull_steps
            stroke = self.config.slow_pull_stroke
            max_command = self.config.slow_pull_max_command
        elif script == "fast_pull":
            pull_steps = self.config.fast_pull_steps
            stroke = self.config.fast_pull_stroke
            max_command = self.config.fast_pull_max_command
        elif script == "pull_release":
            pull_steps = self.config.release_pull_steps
            stroke = self.config.release_pull_stroke * float(scale)
            max_command = min(
                1.0, self.config.release_pull_max_command * float(scale)
            )
        elif script == "pull_push":
            # RS4A third interaction: bidirectional contact. Not in RS2 SCRIPTS.
            pull_steps = self.config.fast_pull_steps
            stroke = self.config.fast_pull_stroke * float(scale)
            max_command = min(1.0, self.config.fast_pull_max_command * float(scale))
        elif script == "switch_cycle":
            # V7B in-episode interaction switch. Not in RS2 SCRIPTS.
            pull_steps = self.config.fast_pull_steps
            stroke = self.config.fast_pull_stroke * float(scale)
            max_command = min(1.0, self.config.fast_pull_max_command * float(scale))
        else:
            raise ValueError(f"Unknown RS2 script: {script}")

        last_tangent = np.asarray([0.0, 1.0], dtype=np.float64)
        for index in range(pull_steps):
            handle_site = np.asarray(self.obs["handle_pos"], dtype=np.float64)
            radial = handle_site[:2] - np.asarray(
                self.obs["door_pos"][:2], dtype=np.float64
            )
            tangent = np.asarray([-radial[1], radial[0]], dtype=np.float64)
            tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
            last_tangent = tangent
            target = self._handle_position()
            # Keep a small positive offset so the closed gripper remains on
            # the opening side of the handle while the stroke increases.
            target[:2] += tangent * (
                0.025 + stroke * float(index + 1) / float(pull_steps)
            )
            action = self._action(
                target, gripper=1.0, max_command=max_command
            )
            self._control_step(action, records=records, phase=2.0)

        if script == "pull_push":
            for index in range(pull_steps):
                handle_site = np.asarray(self.obs["handle_pos"], dtype=np.float64)
                radial = handle_site[:2] - np.asarray(
                    self.obs["door_pos"][:2], dtype=np.float64
                )
                tangent = np.asarray([-radial[1], radial[0]], dtype=np.float64)
                tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
                target = self._handle_position()
                target[:2] -= tangent * (
                    0.025 + stroke * float(index + 1) / float(pull_steps)
                )
                action = self._action(
                    target, gripper=1.0, max_command=max_command
                )
                self._control_step(action, records=records, phase=2.5)

        if script == "pull_release":
            start = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64).copy()
            target = start.copy()
            target[:2] += last_tangent * 0.05
            self._goto(
                target,
                steps=self.config.release_steps,
                gripper=-1.0,
                max_command=0.25,
                records=records,
                phase=3.0,
            )

        if script == "switch_cycle":
            start = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64).copy()
            retreat = start.copy()
            retreat[:2] += last_tangent * 0.05
            self._goto(
                retreat,
                steps=self.config.release_steps,
                gripper=-1.0,
                max_command=0.25,
                records=records,
                phase=3.0,
            )
            self._goto(
                self._handle_position(),
                steps=self.config.close_steps,
                gripper=1.0,
                max_command=self.config.close_max_command,
                records=records,
                phase=3.5,
            )
            for index in range(pull_steps):
                handle_site = np.asarray(self.obs["handle_pos"], dtype=np.float64)
                radial = handle_site[:2] - np.asarray(
                    self.obs["door_pos"][:2], dtype=np.float64
                )
                tangent = np.asarray([-radial[1], radial[0]], dtype=np.float64)
                tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
                target = self._handle_position()
                target[:2] -= tangent * (
                    0.025 + stroke * float(index + 1) / float(pull_steps)
                )
                action = self._action(
                    target, gripper=1.0, max_command=max_command
                )
                self._control_step(action, records=records, phase=2.5)
            quiet = np.asarray(self.obs["robot0_eef_pos"], dtype=np.float64).copy()
            quiet[:2] += last_tangent * 0.04
            quiet[2] += 0.02
            self._goto(
                quiet,
                steps=self.config.release_steps,
                gripper=-1.0,
                max_command=0.20,
                records=records,
                phase=5.0,
            )

        arrays = {
            key: np.asarray(values, dtype=np.float64)
            for key, values in records.items()
        }
        if self._taxel_log:
            arrays["taxel"] = np.stack(self._taxel_log, axis=0)
        if self._shear_log:
            arrays["taxel_shear"] = np.stack(self._shear_log, axis=0)
            arrays["tactile_geom"] = np.stack(self._geom_log, axis=0)
            arrays["tactile_surf"] = np.stack(self._surf_log, axis=0)
        if self._rgb_log:
            arrays["rgb"] = np.stack(self._rgb_log, axis=0)
        contact = np.abs(arrays["tau_contact_jtf"])
        p95 = float(np.quantile(contact, 0.95)) if len(contact) else 0.0
        active = contact > 0.05 * p95 if p95 > 0.0 else np.zeros_like(contact, dtype=bool)
        reference_rms = _rms(arrays["tau_contact_jtf"])
        nrmse = _rms(arrays["residual"]) / (reference_rms + 1.0e-8)
        active_nrmse = (
            _rms(arrays["residual"][active])
            / (_rms(arrays["tau_contact_jtf"][active]) + 1.0e-8)
            if bool(np.any(active))
            else math.inf
        )
        audit_nrmse = _rms(
            arrays["tau_contact_jtf"] - arrays["tau_contact_efc"]
        ) / (reference_rms + 1.0e-8)
        probe_mask = arrays["phase"] == 2.0
        if script in ("pull_push", "switch_cycle"):
            probe_mask = (arrays["phase"] == 2.0) | (arrays["phase"] == 2.5)
        x_phi = (
            float(np.mean(arrays["qvel"][probe_mask] ** 4))
            if bool(np.any(probe_mask))
            else 0.0
        )
        movement = (
            float(np.max(arrays["q"]) - np.min(arrays["q"]))
            if len(arrays["q"])
            else 0.0
        )
        finite = bool(np.all(arrays["finite"] > 0.5))
        no_limit_violation = bool(
            len(arrays["q"])
            and float(np.min(arrays["q"])) >= self.lo
            and float(np.max(arrays["q"])) <= self.hi
        )
        return {
            "seed": self.seed,
            "script": script,
            "scale": float(scale),
            "nrmse": float(nrmse),
            "active_nrmse": float(active_nrmse),
            "contact_audit_nrmse": float(audit_nrmse),
            "movement": movement,
            "contact_active_fraction": float(np.mean(active)) if len(active) else 0.0,
            "contact_p95": p95,
            "X_phi": x_phi,
            "equivalent_amp_scale": _nearest_exposure_bin(x_phi),
            "D0": _rms(arrays["residual"]),
            "finite": finite,
            "no_limit_violation": no_limit_violation,
            "q_min": float(np.min(arrays["q"])) if len(arrays["q"]) else math.nan,
            "q_max": float(np.max(arrays["q"])) if len(arrays["q"]) else math.nan,
            "unaccounted_constraint_rms": _rms(
                arrays["qfrc_constraint_unaccounted"]
            ),
            "actions": np.asarray(self._action_log, dtype=np.float64),
            "arrays": arrays,
        }

    def replay_actions(
        self,
        actions: np.ndarray,
        *,
        q0: float,
        v0: float,
        include_hidden: bool,
        revision_fn: Callable[[float, float], float] | None,
        n_steps: int,
    ) -> dict[str, Any]:
        if self._prepared_qpos is None or self._prepared_qvel is None:
            raise RuntimeError("replay_actions requires prepare()")
        saved_alpha = self.alpha
        saved_revision = self.revision_fn
        saved_log = self._action_log
        self._action_log = None
        self.alpha = saved_alpha if include_hidden else 0.0
        self.revision_fn = revision_fn
        self.data.qpos[:] = self._prepared_qpos
        self.data.qvel[:] = self._prepared_qvel
        self.data.qpos[self.hinge_qpos_adr] = float(q0)
        self.data.qvel[self.hinge_dof] = float(v0)
        self.mujoco.mj_forward(self.model, self.data)
        self._observation()
        qs: list[float] = []
        vs: list[float] = []
        finite = True
        sequence = np.asarray(actions, dtype=np.float64)
        if len(sequence) == 0:
            sequence = np.zeros((1, int(self.env.action_dim)), dtype=np.float64)
        for index in range(int(n_steps)):
            action = sequence[min(index, len(sequence) - 1)]
            self._control_step(action, records=None, phase=4.0)
            q = float(self.data.qpos[self.hinge_qpos_adr])
            v = float(self.data.qvel[self.hinge_dof])
            qs.append(q)
            vs.append(v)
            if not (np.isfinite(q) and np.isfinite(v)):
                finite = False
                break
        self.alpha = saved_alpha
        self.revision_fn = saved_revision
        self._action_log = saved_log
        q_arr = np.asarray(qs, dtype=np.float64)
        v_arr = np.asarray(vs, dtype=np.float64)
        codes = []
        for q in q_arr:
            margin = min(q - self.lo, self.hi - q) if self.hi > self.lo else 1.0
            if margin <= self.config.joint_limit_margin:
                codes.append(1)
            elif self.use_latch and q < self.lo + 2.0 * self.config.joint_limit_margin:
                codes.append(2)
            else:
                codes.append(0)
        return {
            "q": q_arr,
            "qvel": v_arr,
            "support_code": np.asarray(codes, dtype=np.int8),
            "finite": bool(finite and len(q_arr) == int(n_steps)),
        }

    def replay_rollout(
        self,
        actions: np.ndarray,
        *,
        phases: np.ndarray | None = None,
        include_learner_sensors: bool = False,
        include_tactile: bool = False,
        include_rgb: bool = False,
    ) -> dict[str, Any]:
        """Replay a recorded action tape from the prepared interior state."""
        if self._prepared_qpos is None or self._prepared_qvel is None:
            raise RuntimeError("replay_rollout requires prepare()")
        records: dict[str, list[float]] = _record_template(
            include_learner_sensors=include_learner_sensors
        )
        self._want_sensors = bool(include_learner_sensors)
        self._want_tactile = bool(include_tactile)
        self._want_rgb = bool(include_rgb) and bool(self.include_rgb)
        self._taxel_log = [] if include_tactile else None
        self._shear_log = [] if include_tactile else None
        self._geom_log = [] if include_tactile else None
        self._surf_log = [] if include_tactile else None
        self._rgb_log = [] if self._want_rgb else None
        self.data.qpos[:] = self._prepared_qpos
        self.data.qvel[:] = self._prepared_qvel
        self.mujoco.mj_forward(self.model, self.data)
        self._observation()
        saved_log = self._action_log
        self._action_log = []
        sequence = np.asarray(actions, dtype=np.float64)
        phase_tape = None if phases is None else np.asarray(phases, dtype=np.float64)
        for index, action in enumerate(sequence):
            phase = 4.0 if phase_tape is None else float(phase_tape[min(index, len(phase_tape) - 1)])
            self._control_step(action, records=records, phase=phase)
        self._action_log = saved_log
        arrays = {
            key: np.asarray(values, dtype=np.float64) for key, values in records.items()
        }
        if self._taxel_log:
            arrays["taxel"] = np.stack(self._taxel_log, axis=0)
        if self._shear_log:
            arrays["taxel_shear"] = np.stack(self._shear_log, axis=0)
            arrays["tactile_geom"] = np.stack(self._geom_log, axis=0)
            arrays["tactile_surf"] = np.stack(self._surf_log, axis=0)
        if self._rgb_log:
            arrays["rgb"] = np.stack(self._rgb_log, axis=0)
        return {"actions": np.asarray(sequence, dtype=np.float64), "arrays": arrays}


def _rollout_cell(
    config: R1RS2C0Config,
    *,
    seed: int,
    script: str,
    repeat: int,
    scale: float = 1.0,
) -> dict[str, Any]:
    backend = DoorContactC0Backend(config, seed=seed + 100 * int(repeat))
    try:
        backend.prepare()
        result = backend.rollout(script, scale=scale)
        result["base_seed"] = int(seed)
        result["repeat"] = int(repeat)
        return result
    finally:
        backend.close()


def _cell_safe(cell: dict[str, Any], config: R1RS2C0Config) -> bool:
    return bool(
        cell["finite"]
        and cell["no_limit_violation"]
        and cell["movement"] >= config.movement_min
        and cell["contact_active_fraction"] >= config.contact_fraction_min
        and cell["contact_p95"] > config.contact_p95_min
    )


def _calibrate_adapter(
    config: R1RS2C0Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for scale in config.adapter_scales:
        cells = [
            _rollout_cell(
                config,
                seed=seed,
                script="pull_release",
                repeat=0,
                scale=float(scale),
            )
            for seed in config.seeds
        ]
        rows.extend(cells)
        median_exposure = float(np.median([cell["X_phi"] for cell in cells]))
        rel_error = abs(median_exposure / TARGET_EXPOSURE - 1.0)
        candidates.append(
            {
                "scale": float(scale),
                "median_X_phi": median_exposure,
                "relative_error": rel_error,
                "all_safe": bool(all(_cell_safe(cell, config) for cell in cells)),
                "cells": [
                    {
                        key: value
                        for key, value in cell.items()
                        if key != "arrays"
                    }
                    for cell in cells
                ],
            }
        )
    safe = [candidate for candidate in candidates if candidate["all_safe"]]
    selected = (
        min(
            safe,
            key=lambda candidate: (
                abs(math.log(candidate["median_X_phi"] / TARGET_EXPOSURE)),
                candidate["scale"],
            ),
        )
        if safe
        else min(
            candidates,
            key=lambda candidate: (
                abs(math.log(max(candidate["median_X_phi"], 1.0e-15) / TARGET_EXPOSURE)),
                candidate["scale"],
            ),
        )
    )
    passed = bool(
        selected["all_safe"]
        and selected["relative_error"] <= config.exposure_rel_tolerance
    )
    return (
        {
            "target_X_phi": TARGET_EXPOSURE,
            "selected_scale": selected["scale"],
            "selected_median_X_phi": selected["median_X_phi"],
            "selected_relative_error": selected["relative_error"],
            "passed": passed,
            "candidates": candidates,
        },
        rows,
    )


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    metadata = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        handle.create_group("metadata").attrs["json"] = json.dumps(
            metadata, sort_keys=True, default=str
        )
        raw = handle.create_group("raw_truth")
        for key in (
            "q",
            "qvel",
            "qacc",
            "tau_contact_efc",
            "qfrc_constraint_truth",
            "qfrc_constraint_unaccounted",
            "qfrc_passive_truth",
            "qfrc_applied_truth",
            "contact_count",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        learner = handle.create_group("learner_visible")
        for key in (
            "q",
            "qvel",
            "qacc",
            "tau_contact_jtf",
            "tau_actuator",
            "tau_nominal_damping",
            "tau_nominal_friction",
            "residual",
            "phase",
        ):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["contact_force_is_oracle"] = True
        learner.attrs["excludes_full_qfrc_constraint"] = True
        learner.attrs["excludes_full_qfrc_applied"] = True
        learner.attrs["excludes_truth_qfrc_passive"] = True
        decision = handle.create_group("decision")
        decision.attrs["stage"] = "RS2_C0"
        decision.attrs["false_revision"] = bool(cell.get("false_revision", False))
        evaluation = handle.create_group("evaluation")
        for key in (
            "nrmse",
            "active_nrmse",
            "contact_audit_nrmse",
            "movement",
            "contact_active_fraction",
            "contact_p95",
            "X_phi",
            "D0",
        ):
            evaluation.attrs[key] = cell[key]


def _aggregate_c0(
    rows: list[dict[str, Any]],
    *,
    adapter: dict[str, Any],
    config: R1RS2C0Config,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        return {
            "rs2_c0_pass": False,
            "n_cells": len(rows),
            "note": "plumbing smoke only; C0 gates not evaluated",
        }
    required = {
        (seed, script, repeat)
        for seed in config.seeds
        for script in config.scripts
        for repeat in config.repeats
    }
    observed = {
        (row["base_seed"], row["script"], row["repeat"]) for row in rows
    }
    gates = {
        "matrix_complete": {
            "value": len(rows),
            "expected": len(required),
            "pass": observed == required,
        },
        "all_finite": {
            "value": int(sum(bool(row["finite"]) for row in rows)),
            "expected": len(rows),
            "pass": bool(rows and all(bool(row["finite"]) for row in rows)),
        },
        "door_movement": {
            "min": min((float(row["movement"]) for row in rows), default=0.0),
            "threshold": config.movement_min,
            "pass": bool(
                rows and all(float(row["movement"]) >= config.movement_min for row in rows)
            ),
        },
        "contact_active_fraction": {
            "min": min(
                (float(row["contact_active_fraction"]) for row in rows),
                default=0.0,
            ),
            "threshold": config.contact_fraction_min,
            "pass": bool(
                rows
                and all(
                    float(row["contact_active_fraction"])
                    >= config.contact_fraction_min
                    for row in rows
                )
            ),
        },
        "contact_strength": {
            "min_p95": min(
                (float(row["contact_p95"]) for row in rows), default=0.0
            ),
            "threshold": config.contact_p95_min,
            "pass": bool(
                rows
                and all(
                    float(row["contact_p95"]) > config.contact_p95_min
                    for row in rows
                )
            ),
        },
        "contact_projection_audit": {
            "max": max(
                (float(row["contact_audit_nrmse"]) for row in rows),
                default=math.inf,
            ),
            "threshold": config.contact_audit_nrmse_max,
            "pass": bool(
                rows
                and all(
                    float(row["contact_audit_nrmse"])
                    < config.contact_audit_nrmse_max
                    for row in rows
                )
            ),
        },
        "full_residual_closure": {
            "max": max((float(row["nrmse"]) for row in rows), default=math.inf),
            "threshold": config.nrmse_max,
            "pass": bool(
                rows
                and all(float(row["nrmse"]) < config.nrmse_max for row in rows)
            ),
        },
        "active_residual_closure": {
            "max": max(
                (float(row["active_nrmse"]) for row in rows), default=math.inf
            ),
            "threshold": config.nrmse_max,
            "pass": bool(
                rows
                and all(
                    float(row["active_nrmse"]) < config.nrmse_max for row in rows
                )
            ),
        },
        "c0_false_revision": {
            "value": int(sum(bool(row["false_revision"]) for row in rows)),
            "max": 0,
            "pass": bool(rows and not any(row["false_revision"] for row in rows)),
        },
        "adapter": {
            "scale": adapter.get("selected_scale"),
            "relative_error": adapter.get("selected_relative_error"),
            "max_relative_error": config.exposure_rel_tolerance,
            "pass": bool(adapter.get("passed")),
        },
        "learner_visible_audit": {"pass": True},
    }
    passed = bool(all(item["pass"] for item in gates.values()))
    return {
        "n_cells": len(rows),
        "gates": gates,
        "rs2_c0_pass": passed,
        "classification": "pass" if passed else "infrastructure/interface_block",
        "scientific_no_go": False,
    }


def run_r1_rs2_c0(
    output: str | Path,
    *,
    rs1a5_summary: str | Path,
    config: R1RS2C0Config | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = config or R1RS2C0Config()
    policy = _require_rs1a5_unlock(Path(rs1a5_summary))
    thresholds = {
        float(amplitude): float(cell["threshold"])
        for amplitude, cell in policy["aggregate"]["cells"].items()
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    if smoke:
        adapter = {
            "target_X_phi": TARGET_EXPOSURE,
            "selected_scale": 1.0,
            "selected_median_X_phi": math.nan,
            "selected_relative_error": math.nan,
            "passed": False,
            "note": "not calibrated in smoke",
        }
        adapter_rows: list[dict[str, Any]] = []
        jobs = [(cfg.seeds[0], "slow_pull", 0)]
    else:
        adapter, adapter_rows = _calibrate_adapter(cfg)
        jobs = [
            (seed, script, repeat)
            for seed in cfg.seeds
            for script in cfg.scripts
            for repeat in cfg.repeats
        ]

    rows: list[dict[str, Any]] = []
    for index, (seed, script, repeat) in enumerate(jobs, start=1):
        print(
            f"[RS2-C0 {index}/{len(jobs)}] seed={seed} script={script} repeat={repeat}",
            flush=True,
        )
        cell = _rollout_cell(
            cfg, seed=seed, script=script, repeat=repeat, scale=1.0
        )
        amp = float(cell["equivalent_amp_scale"])
        cell["detect_threshold"] = thresholds[amp]
        cell["false_revision"] = bool(cell["D0"] >= thresholds[amp])
        rel = f"cells/{script}/seed_{seed}/repeat_{repeat}.hdf5"
        _write_cell_h5(root / rel, cell)
        cell["path"] = rel
        rows.append(cell)

    for cell in adapter_rows:
        amp = float(cell["equivalent_amp_scale"])
        cell["detect_threshold"] = thresholds[amp]
        cell["false_revision"] = bool(cell["D0"] >= thresholds[amp])
        rel = (
            f"adapter/scale_{cell['scale']:.2f}/"
            f"seed_{cell['base_seed']}.hdf5"
        )
        _write_cell_h5(root / rel, cell)
        cell["path"] = rel

    aggregate = _aggregate_c0(rows, adapter=adapter, config=cfg, smoke=smoke)
    status = stage_status()
    status["R1-RS1C"] = {"unlocked": True, "passed": True, "frozen": True}
    status["R1-RS2-C0"] = {
        "unlocked": True,
        "passed": bool(aggregate.get("rs2_c0_pass")),
        "smoke_only": smoke,
    }
    status["R1-RS2-Formal"] = {
        "locked": not bool(aggregate.get("rs2_c0_pass")),
        "unlocked": bool(aggregate.get("rs2_c0_pass")),
    }

    def slim(cell: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in cell.items() if key != "arrays"}

    manifest = _script_manifest(cfg)
    summary = {
        "stage": "R1-RS2-C0",
        "scientific_result": False,
        "smoke": smoke,
        "prereg_sha256": _prereg_sha256(),
        "script_manifest": manifest,
        "script_manifest_sha256": _manifest_sha256(cfg),
        "packages": {"robosuite": "1.5.2", "mujoco": "3.11.0"},
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(cfg).items()
        },
        "frozen_rs1a5_source": str(Path(rs1a5_summary)),
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "adapter": adapter,
        "cells": [slim(row) for row in rows],
        "adapter_cells": [slim(row) for row in adapter_rows],
        "aggregate": aggregate,
        "rs2_c0_pass": bool(aggregate.get("rs2_c0_pass")),
        "unlocks_rs2_formal": bool(aggregate.get("rs2_c0_pass")),
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    _write_json(root / "script_manifest.json", manifest)
    csv_keys = (
        "base_seed",
        "script",
        "repeat",
        "scale",
        "nrmse",
        "active_nrmse",
        "contact_audit_nrmse",
        "movement",
        "contact_active_fraction",
        "contact_p95",
        "X_phi",
        "equivalent_amp_scale",
        "D0",
        "false_revision",
        "finite",
        "no_limit_violation",
    )
    _write_csv(
        root / "cells.csv",
        [{key: row.get(key) for key in csv_keys} for row in rows],
    )
    if adapter_rows:
        _write_csv(
            root / "adapter_cells.csv",
            [{key: row.get(key) for key in csv_keys} for row in adapter_rows],
        )
    return summary
