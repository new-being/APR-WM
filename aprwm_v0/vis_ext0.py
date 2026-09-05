"""VIS-EXT0: observation-vs-model attribution on robosuite Door visuals.

Raises visual realism only (cluttered Door + Panda). Physics stays Mode-A
1-DoF hinge with known damping mismatch. Does not unlock R10-C0 / RoboCasa.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from scipy import ndimage as ndi

from .mujoco_force import (
    get_bias_force,
    get_constraint_force,
    get_mass_matrix,
    generalized_force_residual,
    nominal_passive_force,
    residual_nrmse,
)
from .vis_ext0_plant import (
    CAMERA_NAME,
    HOST_PLANT_ID,
    HUMAN_PROMPT_BOX_YX,
    HUMAN_PROMPT_CLICK_YX,
    LEARNER_DAMPING,
    TRUE_FRICTION,
    DoorVisPlant,
)
from .vis_x0 import (
    MIN_PANEL_PIXELS,
    _camera_intrinsics,
    _local_linear_slope_and_se,
    _local_mean,
    _reference_q_qd,
    _unwrap_series,
    _write_json,
)
from .vis_x2 import alarm_rate, window_scores
from .vis_x3 import attribution_rates

PREREG_PATH = "REPORT/REG/VISX/VISEXT0_PREREG.md"
SCHEMA_ID = "aprwm.vis_ext0.door_bridge.v1"
NRMSE_ORACLE_MAX = 1.0e-4

ALARM_WINDOW = 50
ALARM_EPS = 1.0e-8
ALARM_QUANTILE = 0.99
U_QUANTILE = 0.99

# Vision-factor constants (frozen before formal; not copied from hinge θ).
TEXTURE_RGB_SCALE = 0.60
TEXTURE_RGB_BIAS = (25.0, -18.0, 40.0)
CAMERA_DELTA_XYZ = (0.04, -0.03, 0.02)
CAMERA_FOVY_DELTA = 3.0
OCCLUSION_FRAC = 0.30
MISMATCH_DAMPING = 0.0

SEGMENTER_ID = "classical_depth_flood_inbox.v1"
SEGMENTER_NOTE = (
    "SAM2 not installed in .venv-robosuite; frozen human first-frame box/click "
    "+ depth-flood propagation preserves the anti-leakage contract "
    "(prompt ≠ GT auto-prompt). GT element seg is eval-only."
)

PhysicsKind = Literal["nominal", "mismatch"]
VisionKind = Literal["clean", "texture_light", "camera", "occlusion", "combined"]
VISION_FACTORS: tuple[VisionKind, ...] = (
    "clean",
    "texture_light",
    "camera",
    "occlusion",
    "combined",
)


@dataclass(frozen=True)
class VISEXT0Config:
    duration_s: float = 1.0
    image_height: int = 128
    image_width: int = 128
    slope_half_window: int = 40
    nrmse_oracle_max: float = NRMSE_ORACLE_MAX
    alarm_window: int = ALARM_WINDOW
    alarm_eps: float = ALARM_EPS
    alarm_quantile: float = ALARM_QUANTILE
    u_quantile: float = U_QUANTILE
    cal_seed: int = 9101
    cal_trajectory: str = "sine"
    test_seeds: tuple[int, ...] = (9111, 9121)
    test_trajectories: tuple[str, ...] = ("sine", "chirp")
    mismatch_damping: float = MISMATCH_DAMPING
    # Door joint range ≈ [0, 0.4]; keep refs inside FOV.
    ref_amplitude: float = 0.12
    ref_center: float = 0.20
    sine_freq_hz: float = 0.50
    chirp_f0_hz: float = 0.15
    chirp_f1_hz: float = 0.50
    piecewise_hold_s: float = 0.50
    pd_kp: float = 120.0
    pd_kd: float = 25.0
    timestep: float = 0.002  # informational; plant dt wins
    require_x3_summary: bool = True
    env_seed: int = 0


class DepthFloodInboxSegmenter:
    """Fixed human box/click → depth-band flood; no GT into the mask."""

    def __init__(
        self,
        *,
        box_yx: tuple[int, int, int, int] = HUMAN_PROMPT_BOX_YX,
        click_yx: tuple[int, int] = HUMAN_PROMPT_CLICK_YX,
        depth_tol: float = 0.030,
        pad: int = 14,
    ):
        self.box_yx = tuple(int(x) for x in box_yx)
        self.click_yx = (int(click_yx[0]), int(click_yx[1]))
        self.depth_tol = float(depth_tol)
        self.pad = int(pad)
        self.d0: float | None = None
        self.mask: np.ndarray | None = None
        self._seed: tuple[int, int] | None = None

    def reset(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float64)
        y0, y1, x0, x1 = self.box_yx
        h, w = depth.shape
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)
        cy, cx = self.click_yx
        cy = int(np.clip(cy, y0, max(y0, y1 - 1)))
        cx = int(np.clip(cx, x0, max(x0, x1 - 1)))
        box = depth[y0:y1, x0:x1]
        finite = box[np.isfinite(box)]
        # Prefer nearer cluster inside the frozen human box (door vs far wall).
        thr = float(np.percentile(finite, 40)) if finite.size else float(depth[cy, cx])
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cand = np.isfinite(depth[yy, xx]) & (depth[yy, xx] <= thr)
        if np.any(cand):
            dist = (yy.astype(np.float64) - cy) ** 2 + (xx.astype(np.float64) - cx) ** 2
            dist = np.where(cand, dist, np.inf)
            k = int(np.argmin(dist))
            sy, sx = int(yy.ravel()[k]), int(xx.ravel()[k])
        else:
            sy, sx = cy, cx
        self._seed = (sy, sx)
        self.d0 = float(depth[sy, sx])
        # First frame: flood inside the human box only (prompt contract).
        self.mask = self._component(depth, sy, sx, roi=(y0, y1, x0, x1))
        return self.mask

    def update(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float64)
        h, w = depth.shape
        if self.mask is None or not np.any(self.mask) or self.d0 is None:
            return self.reset(depth)
        ys, xs = np.where(self.mask)
        cy, cx = int(np.round(ys.mean())), int(np.round(xs.mean()))
        y0 = max(0, int(ys.min()) - self.pad)
        y1 = min(h, int(ys.max()) + self.pad + 1)
        x0 = max(0, int(xs.min()) - self.pad)
        x1 = min(w, int(xs.max()) + self.pad + 1)
        roi_depth = depth[y0:y1, x0:x1]
        near = np.isfinite(roi_depth) & (np.abs(roi_depth - self.d0) <= 0.06)
        if np.any(near):
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist = (yy.astype(np.float64) - cy) ** 2 + (xx.astype(np.float64) - cx) ** 2
            dist = np.where(near, dist, np.inf)
            k = int(np.argmin(dist))
            sy, sx = int(yy.ravel()[k]), int(xx.ravel()[k])
            self.d0 = 0.80 * float(self.d0) + 0.20 * float(depth[sy, sx])
        else:
            sy, sx = cy, cx
        self._seed = (sy, sx)
        self.mask = self._component(depth, sy, sx, roi=(y0, y1, x0, x1))
        if int(np.count_nonzero(self.mask)) < MIN_PANEL_PIXELS:
            # Recover from total loss using frozen human box.
            return self.reset(depth)
        return self.mask

    def _component(
        self,
        depth: np.ndarray,
        sy: int,
        sx: int,
        *,
        roi: tuple[int, int, int, int],
    ) -> np.ndarray:
        y0, y1, x0, x1 = roi
        band = np.isfinite(depth) & (np.abs(depth - float(self.d0)) <= self.depth_tol)
        band = band.copy()
        band[:y0, :] = False
        band[y1:, :] = False
        band[:, :x0] = False
        band[:, x1:] = False
        if not band[sy, sx] and np.isfinite(depth[sy, sx]):
            band[sy, sx] = True
        labeled, nlab = ndi.label(band)
        if nlab <= 0:
            out = np.zeros_like(depth, dtype=bool)
            out[sy, sx] = True
            return out
        lab = int(labeled[sy, sx])
        if lab == 0:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            lab = int(np.argmax(counts)) if counts.size else 0
        if lab == 0:
            out = np.zeros_like(depth, dtype=bool)
            out[sy, sx] = True
            return out
        return labeled == lab


def apply_vision_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    vision: VisionKind,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-render observation ξ (texture/light + occlusion). Camera is plant-side."""

    out_rgb = np.asarray(rgb, dtype=np.uint8).copy()
    out_depth = np.asarray(depth, dtype=np.float64).copy()
    do_tex = vision in ("texture_light", "combined")
    do_occ = vision in ("occlusion", "combined")
    if do_tex:
        tmp = np.asarray(out_rgb, dtype=np.float64) * float(TEXTURE_RGB_SCALE)
        tmp = tmp + np.asarray(TEXTURE_RGB_BIAS, dtype=np.float64)
        out_rgb = np.clip(tmp, 0.0, 255.0).astype(np.uint8)
        # Lighting change couples into depth-band segmentation via depth noise.
        out_depth = out_depth + rng.normal(0.0, 0.025, size=out_depth.shape)
    if do_occ:
        h, w = out_rgb.shape[:2]
        occ_h = max(1, int(round(h * OCCLUSION_FRAC)))
        occ_w = max(1, int(round(w * OCCLUSION_FRAC)))
        y0 = int(rng.integers(0, max(1, h - occ_h + 1)))
        x0 = int(rng.integers(0, max(1, w - occ_w + 1)))
        out_rgb[y0 : y0 + occ_h, x0 : x0 + occ_w] = 0
        out_depth[y0 : y0 + occ_h, x0 : x0 + occ_w] = 1.0e3
    return out_rgb, out_depth


def estimate_q_from_mask_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    plant: DoorVisPlant,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, int]:
    depth = np.asarray(depth, dtype=np.float64)
    mask = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(depth)
        & (depth > 0.05)
        & (depth < 10.0)
    )
    n_pix = int(np.count_nonzero(mask))
    if n_pix < MIN_PANEL_PIXELS:
        return float("nan"), n_pix
    vs, us = np.where(mask)
    zs = depth[vs, us]
    xs = (us.astype(np.float64) - cx) / fx * zs
    ys = (vs.astype(np.float64) - cy) / fy * zs
    pts_cam = np.stack([xs, ys, -zs], axis=1)
    R = np.asarray(plant.data.cam_xmat[plant.camera_id], dtype=np.float64).reshape(3, 3)
    p = np.asarray(plant.data.cam_xpos[plant.camera_id], dtype=np.float64)
    pts_w = (R @ pts_cam.T).T + p
    pivot = plant.hinge_pivot_world()
    xy = pts_w[:, :2] - pivot[:2]
    # Centroid bearing is more stable than PCA under partial masks on Door.
    c = xy.mean(axis=0)
    if float(np.linalg.norm(c)) < 1.0e-6:
        return float("nan"), n_pix
    return float(np.arctan2(c[1], c[0])), n_pix


def _door_reference_q_qd(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    config: VISEXT0Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift VIS-X style refs into Door joint range."""

    # Temporarily build a VISX0-compatible shim for _reference_q_qd.
    from .vis_x0 import VISX0Config

    shim = VISX0Config(
        duration_s=config.duration_s,
        timestep=dt,
        ref_amplitude=config.ref_amplitude,
        sine_freq_hz=config.sine_freq_hz,
        chirp_f0_hz=config.chirp_f0_hz,
        chirp_f1_hz=config.chirp_f1_hz,
        piecewise_hold_s=config.piecewise_hold_s,
        pd_kp=config.pd_kp,
        pd_kd=config.pd_kd,
        slope_half_window=config.slope_half_window,
    )
    q_ref, qd_ref = _reference_q_qd(
        kind, n_steps=n_steps, dt=dt, seed=seed, config=shim
    )
    q_ref = np.clip(q_ref + float(config.ref_center), 0.02, 0.38)
    return q_ref, qd_ref


def _visual_residual_hinge(
    plant: DoorVisPlant,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    qacc: np.ndarray,
    tau: np.ndarray,
    constraint_oracle: np.ndarray | None = None,
) -> np.ndarray:
    """Perception residual on the hinge DoF.

    Multi-body constraint forces are taken from the oracle rollout step
    (contact / joint forces of the plant), not recomputed at the visual
    configuration — recomputation spuriously spikes when the estimated door
    pose slightly intersects the frozen Panda. Perception enters only via
    ``q`` / ``qd`` (mass, bias, nominal damping).
    """

    out = np.zeros(q.shape[0], dtype=np.float64)
    dof = plant.hinge_dof
    if constraint_oracle is None:
        c_log = np.zeros(q.shape[0], dtype=np.float64)
    else:
        c_log = np.asarray(constraint_oracle, dtype=np.float64).reshape(-1)
    for i in range(q.shape[0]):
        plant.set_state(float(q[i]), float(qd[i]))
        mass = get_mass_matrix(plant.model, plant.data)
        bias = get_bias_force(plant.data)
        passive = nominal_passive_force(
            np.asarray(plant.data.qvel, dtype=np.float64),
            plant.nominal,
            dof_index=dof,
        )
        qacc_vec = np.zeros(plant.model.nv, dtype=np.float64)
        qacc_vec[dof] = float(qacc[i])
        out[i] = float(
            (mass @ qacc_vec)[dof]
            + float(bias[dof])
            - float(tau[i])
            - float(c_log[i])
            - float(passive[dof])
        )
    return out


def _mask_iou(mask: np.ndarray, seg: np.ndarray, panel_geom_id: int) -> float:
    gt = np.asarray(seg[..., 0] if seg.ndim == 3 else seg, dtype=np.int32) == int(
        panel_geom_id
    )
    m = np.asarray(mask, dtype=bool)
    inter = float(np.count_nonzero(m & gt))
    union = float(np.count_nonzero(m | gt))
    return float(inter / union) if union > 0 else 0.0


def _load_x3_passed(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"VIS-EXT0 requires VIS-X3 summary at {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("vis_x3_passed")):
        raise RuntimeError("VIS-EXT0 locked until vis_x3_passed=true")
    return payload


def _rollout_cell(
    plant: DoorVisPlant,
    config: VISEXT0Config,
    *,
    seed: int,
    trajectory: str,
    physics: PhysicsKind,
    vision: VisionKind,
    true_damping: float,
    role: str,
) -> dict[str, Any]:
    plant.set_true_passive(friction=TRUE_FRICTION, damping=true_damping)
    plant.reset_camera()
    if vision in ("camera", "combined"):
        plant.apply_camera_shift(CAMERA_DELTA_XYZ, fovy_delta=CAMERA_FOVY_DELTA)

    dt = float(plant.dt)
    n_steps = int(round(config.duration_s / dt))
    q_ref, qd_ref = _door_reference_q_qd(
        trajectory, n_steps=n_steps, dt=dt, seed=seed, config=config
    )
    fx, fy, cx, cy = _camera_intrinsics(
        config.image_height, config.image_width, plant.fovy
    )
    rng = np.random.default_rng(
        int(seed)
        + 17 * abs(int.from_bytes(trajectory.encode(), "little") % 10_000)
        + 101 * abs(int(round(true_damping * 1000.0)))
        + 1009 * (0 if physics == "nominal" else 1)
        + 2003 * VISION_FACTORS.index(vision)
    )
    segmenter = DepthFloodInboxSegmenter()

    q_oracle: list[float] = []
    qd_oracle: list[float] = []
    qacc_plant: list[float] = []
    constraint_oracle: list[float] = []
    q_hat_raw: list[float] = []
    n_panel: list[int] = []
    residual_oracle: list[float] = []
    tau_log: list[float] = []
    iou_log: list[float] = []

    q0 = float(config.ref_center)
    plant.set_state(q0, 0.0)
    rgb0, depth0, seg0 = plant.observe_rgbd_gt()
    rgb0, depth0 = apply_vision_rgb_depth(rgb0, depth0, vision=vision, rng=rng)
    mask0 = segmenter.reset(depth0)
    ang0, _ = estimate_q_from_mask_depth(
        depth0, mask0, plant=plant, fx=fx, fy=fy, cx=cx, cy=cy
    )
    if not np.isfinite(ang0):
        ang0 = 0.0

    for i in range(n_steps):
        plant._freeze_robot()
        if plant.model.nu:
            plant.data.ctrl[:] = 0.0
        qpos_pre = float(plant.data.qpos[plant.hinge_qpos_adr])
        qvel_pre = float(plant.data.qvel[plant.hinge_dof])
        qvel_vec_pre = np.array(plant.data.qvel, dtype=np.float64, copy=True)

        rgb, depth, seg = plant.observe_rgbd_gt()
        rgb, depth = apply_vision_rgb_depth(rgb, depth, vision=vision, rng=rng)
        mask = mask0 if i == 0 else segmenter.update(depth)
        if i == 0:
            mask = mask0
        ang, n_pix = estimate_q_from_mask_depth(
            depth, mask, plant=plant, fx=fx, fy=fy, cx=cx, cy=cy
        )
        if np.isfinite(ang):
            q_est = float(ang - ang0) + q0
        else:
            q_est = float("nan")

        tau = float(
            config.pd_kp * (q_ref[i] - qpos_pre)
            + config.pd_kd * (qd_ref[i] - qvel_pre)
        )
        plant.data.qfrc_applied[:] = 0.0
        if plant.model.nu:
            plant.data.ctrl[:] = 0.0
        plant.data.qfrc_applied[plant.hinge_dof] = tau
        plant.mujoco.mj_step(plant.model, plant.data)

        residual = generalized_force_residual(
            plant.model,
            plant.data,
            nominal_passive=plant.nominal,
            include_constraint=True,
            dof_index=plant.hinge_dof,
            qvel_for_passive=qvel_vec_pre,
        )
        c_h = float(get_constraint_force(plant.data)[plant.hinge_dof])
        q_oracle.append(qpos_pre)
        qd_oracle.append(qvel_pre)
        qacc_plant.append(float(plant.data.qacc[plant.hinge_dof]))
        constraint_oracle.append(c_h)
        q_hat_raw.append(q_est)
        n_panel.append(int(n_pix))
        residual_oracle.append(float(np.asarray(residual, dtype=np.float64)[plant.hinge_dof]))
        tau_log.append(tau)
        iou_log.append(_mask_iou(mask, seg, plant.panel_geom_id))
        plant._freeze_robot()
        plant.mujoco.mj_forward(plant.model, plant.data)

    plant.reset_camera()

    q_o = np.asarray(q_oracle, dtype=np.float64)
    qd_o = np.asarray(qd_oracle, dtype=np.float64)
    qacc = np.asarray(qacc_plant, dtype=np.float64)
    c_o = np.asarray(constraint_oracle, dtype=np.float64)
    tau_a = np.asarray(tau_log, dtype=np.float64)
    r_o = np.asarray(residual_oracle, dtype=np.float64)
    q_hat = _unwrap_series(np.asarray(q_hat_raw, dtype=np.float64))
    half = int(config.slope_half_window)
    q_hat_s = _local_mean(q_hat, half)
    qd_hat, u_hat = _local_linear_slope_and_se(q_hat, dt, half)
    r_v = _visual_residual_hinge(
        plant,
        q=q_hat_s,
        qd=qd_hat,
        qacc=qacc,
        tau=tau_a,
        constraint_oracle=c_o,
    )
    r_pseudo = r_v - r_o

    e_q = float(np.sqrt(np.mean(np.square(q_hat_s - q_o))))
    e_qd = float(np.sqrt(np.mean(np.square(qd_hat - qd_o))))
    e_r = float(residual_nrmse(r_v, tau_a))
    e_pseudo = float(np.sqrt(np.mean(np.square(r_pseudo))))
    nrmse_o = float(residual_nrmse(r_o, tau_a))
    scores = window_scores(
        r_v, tau_a, window=config.alarm_window, eps=config.alarm_eps
    )
    finite_ok = bool(np.all(np.isfinite(q_hat_s)) and np.all(np.isfinite(qd_hat)))
    min_pix = int(min(n_panel)) if n_panel else 0
    g_phys_clean = bool(nrmse_o < config.nrmse_oracle_max) if physics == "nominal" else True
    # Hard ξ may legitimately lose the door; require pixels only on clean vision.
    if vision == "clean":
        g_run = bool(finite_ok and min_pix >= max(10, MIN_PANEL_PIXELS // 3))
    else:
        g_run = bool(finite_ok)

    return {
        "role": role,
        "seed": seed,
        "trajectory": trajectory,
        "physics": physics,
        "vision": vision,
        "true_damping": float(true_damping),
        "learner_damping": float(LEARNER_DAMPING),
        "host_plant_id": HOST_PLANT_ID,
        "camera": CAMERA_NAME,
        "segmenter_id": SEGMENTER_ID,
        "n_steps": n_steps,
        "E_q": e_q,
        "E_qd": e_qd,
        "E_r": e_r,
        "E_pseudo": e_pseudo,
        "nrmse_oracle": nrmse_o,
        "mean_mask_iou": float(np.mean(iou_log)) if iou_log else float("nan"),
        "min_panel_pixels": min_pix,
        "g_phys_clean": g_phys_clean,
        "g_run": g_run,
        "scores": scores,
        "U": u_hat,
        "arrays": {
            "q_oracle": q_o,
            "qd_oracle": qd_o,
            "q_hat": q_hat_s,
            "qd_hat": qd_hat,
            "U": u_hat,
            "tau": tau_a,
            "residual_oracle": r_o,
            "residual_visual": r_v,
            "residual_pseudo": r_pseudo,
            "alarm_scores": scores,
            "n_panel_pixels": np.asarray(n_panel, dtype=np.int32),
            "mask_iou": np.asarray(iou_log, dtype=np.float64),
        },
    }


def _write_cell_h5(
    path: Path,
    cell: dict[str, Any],
    *,
    theta: float,
    tau_u: float,
    rates: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {k: v for k, v in cell.items() if k not in ("arrays", "scores", "U")}
    meta["theta"] = float(theta)
    meta["tau_U"] = float(tau_u)
    if rates:
        meta.update(rates)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "vis-ext0"
        handle.attrs["host_plant_id"] = HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["real_perception"] = False
        handle.attrs["unlocks_r10_c0"] = False
        handle.attrs["physics"] = cell["physics"]
        handle.attrs["vision"] = cell["vision"]
        handle.attrs["true_damping"] = cell["true_damping"]
        meta_grp = handle.create_group("metadata")
        meta_grp.attrs["json"] = json.dumps(meta, sort_keys=True, default=str)
        learner = handle.create_group("learner_visible")
        for key in ("q_hat", "qd_hat", "U", "tau", "residual_visual", "alarm_scores"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["theta"] = float(theta)
        learner.attrs["tau_U"] = float(tau_u)
        raw = handle.create_group("raw_truth")
        for key in ("q_oracle", "qd_oracle", "residual_oracle", "n_panel_pixels", "mask_iou"):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        raw.attrs["audit_only"] = True
        if rates:
            evaluation = handle.create_group("evaluation")
            for key, value in rates.items():
                evaluation.attrs[key] = value


def run_vis_ext0(
    output: str | Path,
    *,
    config: VISEXT0Config | None = None,
    x3_summary: str | Path | None = "runs/vis_x3/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or VISEXT0Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("VIS-EXT0 must not write under runs/r10_c0/")
    x3 = None
    if cfg.require_x3_summary:
        x3 = _load_x3_passed(x3_summary)
    root.mkdir(parents=True, exist_ok=True)

    plant = DoorVisPlant(
        seed=cfg.env_seed,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        true_damping=LEARNER_DAMPING,
    )
    try:
        inventory = plant.inventory()
        _write_json(root / "plant_inventory.json", inventory)
        _write_json(
            root / "alarm_harness.json",
            {
                "window": cfg.alarm_window,
                "eps": cfg.alarm_eps,
                "quantile": cfg.alarm_quantile,
                "u_quantile": cfg.u_quantile,
                "note": "Recalibrated on Door nominal+clean; do not copy hinge θ/τ_U",
                "segmenter_id": SEGMENTER_ID,
                "segmenter_note": SEGMENTER_NOTE,
                "vision_factors": list(VISION_FACTORS),
                "texture_rgb_scale": TEXTURE_RGB_SCALE,
                "texture_rgb_bias": list(TEXTURE_RGB_BIAS),
                "camera_delta_xyz": list(CAMERA_DELTA_XYZ),
                "camera_fovy_delta": CAMERA_FOVY_DELTA,
                "occlusion_frac": OCCLUSION_FRAC,
                "mismatch_damping": cfg.mismatch_damping,
                "retune_after_formal_forbidden": True,
            },
        )

        # --- Calibration: nominal + clean ---
        cal = _rollout_cell(
            plant,
            cfg,
            seed=cfg.cal_seed,
            trajectory=cfg.cal_trajectory,
            physics="nominal",
            vision="clean",
            true_damping=LEARNER_DAMPING,
            role="calibration",
        )
        if not cal["g_phys_clean"]:
            raise RuntimeError(
                f"VIS-EXT0 G0 failed on calibration: nrmse_oracle={cal['nrmse_oracle']}"
            )
        if not cal["g_run"]:
            raise RuntimeError("VIS-EXT0 calibration cell failed g_run")
        theta = float(np.quantile(cal["scores"], cfg.alarm_quantile))
        u_cal = np.asarray(cal["U"], dtype=np.float64)
        u_cal = u_cal[np.isfinite(u_cal)]
        tau_u = float(np.quantile(u_cal, cfg.u_quantile))
        _write_cell_h5(
            root / "calibration" / f"seed_{cfg.cal_seed}_{cfg.cal_trajectory}.h5",
            cal,
            theta=theta,
            tau_u=tau_u,
        )

        specs: list[tuple[PhysicsKind, VisionKind, float]] = []
        for vision in VISION_FACTORS:
            specs.append(("nominal", vision, LEARNER_DAMPING))
            specs.append(("mismatch", vision, float(cfg.mismatch_damping)))

        rows: list[dict[str, Any]] = []
        for seed in cfg.test_seeds:
            for trajectory in cfg.test_trajectories:
                for physics, vision, b_true in specs:
                    cell = _rollout_cell(
                        plant,
                        cfg,
                        seed=seed,
                        trajectory=trajectory,
                        physics=physics,
                        vision=vision,
                        true_damping=b_true,
                        role="test",
                    )
                    rates = attribution_rates(
                        cell["scores"],
                        cell["U"],
                        theta=theta,
                        tau_u=tau_u,
                        window=cfg.alarm_window,
                    )
                    key = f"{physics}_{vision}"
                    rel = f"test/seed_{seed}/{trajectory}/{key}.h5"
                    _write_cell_h5(
                        root / rel, cell, theta=theta, tau_u=tau_u, rates=rates
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "trajectory": trajectory,
                            "physics": physics,
                            "vision": vision,
                            "true_damping": b_true,
                            "condition": key,
                            "path": str(root / rel),
                            "E_q": cell["E_q"],
                            "E_qd": cell["E_qd"],
                            "E_r": cell["E_r"],
                            "E_pseudo": cell["E_pseudo"],
                            "nrmse_oracle": cell["nrmse_oracle"],
                            "mean_mask_iou": cell["mean_mask_iou"],
                            "alarm_rate": rates["alarm_rate"],
                            "phyclaim_rate": rates["phyclaim_rate"],
                            "ambiguous_rate": rates["ambiguous_rate"],
                            "d_rate_steps": rates["d_rate_steps"],
                            "g_phys_clean": cell["g_phys_clean"],
                            "g_run": cell["g_run"],
                            "min_panel_pixels": cell["min_panel_pixels"],
                        }
                    )
    finally:
        plant.close()

    def _mean(physics: str, vision: str, key: str) -> float:
        vals = [
            float(r[key])
            for r in rows
            if r["physics"] == physics and r["vision"] == vision
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_visual_group(physics: str, key: str, visions: tuple[str, ...]) -> float:
        vals = [
            float(r[key])
            for r in rows
            if r["physics"] == physics and r["vision"] in visions
        ]
        return float(np.mean(vals)) if vals else float("nan")

    hard_vis: tuple[str, ...] = ("occlusion", "combined")
    any_vis: tuple[str, ...] = tuple(v for v in VISION_FACTORS if v != "clean")

    fpr_clean = _mean("nominal", "clean", "alarm_rate")
    fpr_visual = _mean_visual_group("nominal", "alarm_rate", any_vis)
    fpr_hard = _mean_visual_group("nominal", "alarm_rate", hard_vis)
    tpr_alarm = _mean("mismatch", "clean", "alarm_rate")
    fpr_phyclaim_visual = _mean_visual_group("nominal", "phyclaim_rate", any_vis)
    tpr_phyclaim = _mean("mismatch", "clean", "phyclaim_rate")

    e_qd_clean = _mean("nominal", "clean", "E_qd")
    e_qd_hard = _mean_visual_group("nominal", "E_qd", hard_vis)
    e_pseudo_clean = _mean("nominal", "clean", "E_pseudo")
    e_pseudo_hard = _mean_visual_group("nominal", "E_pseudo", hard_vis)

    nominal_rows = [r for r in rows if r["physics"] == "nominal"]
    g0 = all(bool(r["g_phys_clean"]) for r in nominal_rows)
    g1 = bool(e_qd_hard > e_qd_clean and e_pseudo_hard > e_pseudo_clean)
    g2 = bool(fpr_visual > fpr_clean)
    g3 = bool(
        np.isfinite(fpr_phyclaim_visual)
        and np.isfinite(fpr_visual)
        and fpr_phyclaim_visual <= 0.5 * fpr_visual
    )
    g4 = bool(
        np.isfinite(tpr_phyclaim)
        and np.isfinite(tpr_alarm)
        and tpr_alarm >= 0.05
        and tpr_phyclaim >= 0.8 * tpr_alarm
    )
    g_run = all(bool(r["g_run"]) for r in rows)
    passed = bool(g0 and g1 and g2 and g3 and g4 and g_run)

    if g0 and g1 and g2 and g3 and g4:
        outcome = "attribution_external_validity"
    elif not g0:
        outcome = "oracle_closure_fail"
    elif g3 and not g4:
        outcome = "over_veto"
    elif g2 and not g3:
        outcome = "masquerade_unrescued"
    else:
        outcome = "mixed_or_partial"

    summary = {
        "stage": "VIS-EXT0",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": (
            "observation-vs-model attribution survives a more realistic, "
            "cluttered robot-manipulation visual distribution"
            if passed
            else "VIS-EXT0 gates not all satisfied; no external-validity upgrade"
        ),
        "real_physics": False,
        "real_perception": False,
        "unlocks_r10_c0": False,
        "unlocks_robocasa": bool(passed),
        "segmenter_id": SEGMENTER_ID,
        "segmenter_note": SEGMENTER_NOTE,
        "x3_summary": str(x3_summary) if x3_summary else None,
        "x3_passed": None if x3 is None else bool(x3.get("vis_x3_passed")),
        "config": asdict(cfg),
        "alarm_harness": {
            "window": cfg.alarm_window,
            "eps": cfg.alarm_eps,
            "quantile": cfg.alarm_quantile,
            "u_quantile": cfg.u_quantile,
            "theta": theta,
            "tau_U": tau_u,
            "calibration": {
                "seed": cfg.cal_seed,
                "trajectory": cfg.cal_trajectory,
                "physics": "nominal",
                "vision": "clean",
                "n_scores": int(cal["scores"].size),
            },
            "copied_from_hinge": False,
            "retune_after_formal_forbidden": True,
        },
        "metrics": {
            "FPR_alarm_clean": fpr_clean,
            "FPR_alarm_visual": fpr_visual,
            "FPR_alarm_hard": fpr_hard,
            "TPR_alarm_clean_mismatch": tpr_alarm,
            "FPR_phyclaim_visual": fpr_phyclaim_visual,
            "TPR_phyclaim_clean_mismatch": tpr_phyclaim,
            "mean_E_qd_nominal_clean": e_qd_clean,
            "mean_E_qd_nominal_hard": e_qd_hard,
            "mean_E_pseudo_nominal_clean": e_pseudo_clean,
            "mean_E_pseudo_nominal_hard": e_pseudo_hard,
        },
        "gates": {
            "G0_physics_closure": g0,
            "G1_realistic_challenge": g1,
            "G2_masquerade_replication": g2,
            "G3_attribution_rescue": g3,
            "G4_physics_retention": g4,
            "g_run": g_run,
        },
        "outcome_pattern": outcome,
        "cells": rows,
        "cell_count": len(rows),
        "vis_ext0_passed": passed,
        "pass_iff": (
            "G0 NRMSE(r_oracle)<1e-4 on nominal-P; G1 hard ξ raises E_qd & E_pseudo; "
            "G2 FPR_alarm visual>clean; G3 FPR_phyclaim≤0.5 FPR_alarm; "
            "G4 TPR_phyclaim≥0.8 TPR_alarm on clean mismatch"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
