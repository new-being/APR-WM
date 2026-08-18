"""VIS-X0: GT segmentation + depth → state → force-residual interface smoke.

Does not unlock R10-C0. Does not train detectors. Simulator seg ≠ real vision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import (
    generalized_force_residual,
    get_bias_force,
    get_mass_matrix,
    nominal_passive_force,
    residual_nrmse,
)
from .mujoco_physics import import_mujoco_with_renderer
from .simx_plant import learner_nominal_params
from .simx_vis_plant import (
    VIS_CAMERA_NAME,
    VIS_HOST_PLANT_ID,
    VIS_PANEL_GEOM,
    VIS_PIVOT_XYZ,
    assert_vis_plant,
    load_simx_hinge_vis,
    simx_hinge_vis_xml,
    simx_hinge_vis_xml_sha256,
)

PREREG_PATH = "REPORT/REG/VISX/VISX0_PREREG.md"
SCHEMA_ID = "aprwm.vis_x0.residual.v1"

VISX0_SEEDS = (9101, 9111, 9121)
VISX0_TRAJECTORIES = ("sine", "chirp", "piecewise")

# Floors frozen after first camera campaign (convention gates).
Q_RMSE_MAX = 0.05
NRMSE_ORACLE_MAX = 1.0e-4
# Visual proprio (q̂, q̂̇) + plant qacc — isolates perception in residual channels.
NRMSE_VISUAL_MAX = 0.10
MIN_PANEL_PIXELS = 30


@dataclass(frozen=True)
class VISX0Config:
    seeds: tuple[int, ...] = VISX0_SEEDS
    trajectories: tuple[str, ...] = VISX0_TRAJECTORIES
    duration_s: float = 4.0
    timestep: float = 0.002
    image_height: int = 240
    image_width: int = 320
    # Local linear half-window (samples) for q smooth / qdot slope.
    slope_half_window: int = 40
    q_rmse_max: float = Q_RMSE_MAX
    nrmse_oracle_max: float = NRMSE_ORACLE_MAX
    nrmse_visual_max: float = NRMSE_VISUAL_MAX
    # PD tracks references inside the fixed-camera FOV (oracle used only for drive).
    ref_amplitude: float = 0.80
    sine_freq_hz: float = 0.35
    chirp_f0_hz: float = 0.10
    chirp_f1_hz: float = 0.70
    piecewise_hold_s: float = 0.50
    pd_kp: float = 8.0
    pd_kd: float = 2.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _wrap_angle(delta: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(delta, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _reference_q_qd(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    config: VISX0Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded joint references for PD drive (not the perception observation)."""

    t = np.arange(n_steps, dtype=np.float64) * dt
    amp = float(config.ref_amplitude)
    if kind == "sine":
        w = 2.0 * np.pi * float(config.sine_freq_hz)
        q_ref = amp * np.sin(w * t)
        qd_ref = amp * w * np.cos(w * t)
        return q_ref, qd_ref
    if kind == "chirp":
        f0 = float(config.chirp_f0_hz)
        f1 = float(config.chirp_f1_hz)
        duration = max(float(config.duration_s), dt)
        k = (f1 - f0) / duration
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        q_ref = amp * np.sin(phase)
        # d/dt sin(φ) = cos(φ) φ'; φ' = 2π (f0 + k t)
        qd_ref = amp * np.cos(phase) * (2.0 * np.pi * (f0 + k * t))
        return q_ref, qd_ref
    if kind == "piecewise":
        rng = np.random.default_rng(seed + 17_000)
        hold = max(1, int(round(config.piecewise_hold_s / dt)))
        levels: list[float] = []
        while len(levels) < n_steps:
            levels.extend([float(rng.uniform(-amp, amp))] * hold)
        q_ref = np.asarray(levels[:n_steps], dtype=np.float64)
        qd_ref = np.gradient(q_ref, dt)
        return q_ref, qd_ref
    raise ValueError(f"Unknown trajectory kind: {kind}")


def _local_mean(y: np.ndarray, half: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    half = max(1, int(half))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def _local_linear_slope(y: np.ndarray, dt: float, half: int) -> np.ndarray:
    """Centered local-linear slope (Savitzky–Golay-like, pure NumPy)."""

    slope, _se = _local_linear_slope_and_se(y, dt, half)
    return slope


def _local_linear_slope_and_se(
    y: np.ndarray, dt: float, half: int
) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear slope and learner-visible slope standard error.

    Fits ``q_j = α + β(t_j - t) + e_j`` on a centered window and returns
    ``β̂`` with ``SE(β̂) = sqrt(σ̂_e² / Σ(t_j - t̄)²)``.
    """

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    half = max(1, int(half))
    slope = np.empty(n, dtype=np.float64)
    se = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xx = (np.arange(lo, hi, dtype=np.float64) - float(i)) * float(dt)
        yy = y[lo:hi]
        n_pts = int(yy.size)
        yy_c = yy - float(np.mean(yy))
        denom = float(np.dot(xx, xx))
        if denom <= 1.0e-18 or n_pts < 3:
            slope[i] = 0.0
            se[i] = float("inf")
            continue
        beta = float(np.dot(xx, yy_c) / denom)
        resid = yy_c - beta * xx
        sigma2 = float(np.dot(resid, resid) / float(n_pts - 2))
        slope[i] = beta
        se[i] = float(np.sqrt(max(sigma2, 0.0) / denom))
    return slope, se


def _camera_intrinsics(height: int, width: int, fovy_deg: float) -> tuple[float, float, float, float]:
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    fx = fy
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    return float(fx), float(fy), float(cx), float(cy)


def estimate_q_from_seg_depth(
    seg: np.ndarray,
    depth: np.ndarray,
    *,
    model: Any,
    data: Any,
    panel_geom_id: int,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, int]:
    """Pivot-centered PCA on panel depth points → hinge angle.

    Uses simulator GT segmentation (geom id channel). Not real perception.
    """

    depth = np.asarray(depth, dtype=np.float64)
    geom = np.asarray(seg[..., 0], dtype=np.int32)
    mask = (
        (geom == int(panel_geom_id))
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
    # MuJoCo camera looks along -Z in camera frame.
    pts_cam = np.stack([xs, ys, -zs], axis=1)
    R = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    p = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    pts_w = (R @ pts_cam.T).T + p
    pivot = np.asarray(VIS_PIVOT_XYZ, dtype=np.float64)
    xy = pts_w[:, :2] - pivot[:2]
    cov = xy.T @ xy
    _, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, -1]
    if float(axis @ xy.mean(axis=0)) < 0.0:
        axis = -axis
    return float(np.arctan2(axis[1], axis[0])), n_pix


def _unwrap_series(raw: np.ndarray) -> np.ndarray:
    """Unwrap hinge angles; resolve PCA ±π flips by min-step continuity."""

    src = np.asarray(raw, dtype=np.float64)
    out = np.zeros(src.shape[0], dtype=np.float64)
    if src.size == 0:
        return out
    out[0] = float(src[0]) if np.isfinite(src[0]) else 0.0
    for i in range(1, src.size):
        if not np.isfinite(src[i]):
            out[i] = out[i - 1]
            continue
        prev = float(out[i - 1])
        prev_wrapped = float(_wrap_angle(prev))
        d0 = float(_wrap_angle(src[i] - prev_wrapped))
        d1 = float(_wrap_angle(src[i] + np.pi - prev_wrapped))
        step = d1 if abs(d1) < abs(d0) else d0
        out[i] = prev + step
    return out


def _render_seg_depth(renderer: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=VIS_CAMERA_NAME)
    renderer.enable_segmentation_rendering()
    seg = np.array(renderer.render(), copy=True)
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    depth = np.array(renderer.render(), copy=True)
    renderer.disable_depth_rendering()
    return seg, depth


def _visual_residual_stream(
    model: Any,
    data: Any,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    qacc: np.ndarray,
    tau: np.ndarray,
    nominal: Any,
) -> np.ndarray:
    mujoco = import_mujoco_with_renderer()
    out = np.zeros(q.shape[0], dtype=np.float64)
    for i in range(q.shape[0]):
        data.qpos[0] = float(q[i])
        data.qvel[0] = float(qd[i])
        mujoco.mj_forward(model, data)
        mass = get_mass_matrix(model, data)
        bias = get_bias_force(data)
        passive = nominal_passive_force(
            np.asarray([qd[i]], dtype=np.float64), nominal, dof_index=0
        )
        known = np.asarray([tau[i]], dtype=np.float64)
        out[i] = float((mass @ np.asarray([qacc[i]]) + bias - known - passive)[0])
    return out


def _rollout_cell(config: VISX0Config, *, seed: int, trajectory: str) -> dict[str, Any]:
    mujoco = import_mujoco_with_renderer()
    model, data = load_simx_hinge_vis(config.timestep)
    inventory = assert_vis_plant(model)
    panel_id = int(inventory["panel_geom_id"])
    cam_id = int(inventory["camera_id"])
    fovy = float(model.cam_fovy[cam_id])
    fx, fy, cx, cy = _camera_intrinsics(config.image_height, config.image_width, fovy)

    n_steps = int(round(config.duration_s / config.timestep))
    q_ref, qd_ref = _reference_q_qd(
        trajectory,
        n_steps=n_steps,
        dt=config.timestep,
        seed=seed,
        config=config,
    )
    nominal = learner_nominal_params()
    renderer = mujoco.Renderer(model, height=config.image_height, width=config.image_width)

    q_oracle: list[float] = []
    qd_oracle: list[float] = []
    qacc_plant: list[float] = []
    q_hat_raw: list[float] = []
    n_panel: list[int] = []
    residual_oracle: list[float] = []
    tau_log: list[float] = []

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.15 * np.sin(seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    try:
        for i in range(n_steps):
            seg, depth = _render_seg_depth(renderer, data)
            q_est, n_pix = estimate_q_from_seg_depth(
                seg,
                depth,
                model=model,
                data=data,
                panel_geom_id=panel_id,
                camera_id=cam_id,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            qpos_pre = np.array(data.qpos, dtype=np.float64, copy=True)
            qvel_pre = np.array(data.qvel, dtype=np.float64, copy=True)
            # Oracle state used only to synthesize FOV-safe excitation.
            tau = float(
                config.pd_kp * (q_ref[i] - qpos_pre[0])
                + config.pd_kd * (qd_ref[i] - qvel_pre[0])
            )
            data.qfrc_applied[:] = 0.0
            data.qfrc_actuator[:] = 0.0
            if model.nu:
                data.ctrl[:] = 0.0
            data.qfrc_applied[0] = tau
            mujoco.mj_step(model, data)
            residual = generalized_force_residual(
                model,
                data,
                nominal_passive=nominal,
                include_constraint=True,
                dof_index=0,
                qvel_for_passive=qvel_pre,
            )
            q_oracle.append(float(qpos_pre[0]))
            qd_oracle.append(float(qvel_pre[0]))
            qacc_plant.append(float(np.asarray(data.qacc, dtype=np.float64)[0]))
            q_hat_raw.append(float(q_est))
            n_panel.append(int(n_pix))
            residual_oracle.append(float(np.asarray(residual, dtype=np.float64)[0]))
            tau_log.append(tau)
    finally:
        renderer.close()

    q_o = np.asarray(q_oracle, dtype=np.float64)
    qd_o = np.asarray(qd_oracle, dtype=np.float64)
    qacc = np.asarray(qacc_plant, dtype=np.float64)
    tau_a = np.asarray(tau_log, dtype=np.float64)
    r_o = np.asarray(residual_oracle, dtype=np.float64)
    q_hat = _unwrap_series(np.asarray(q_hat_raw, dtype=np.float64))
    half = int(config.slope_half_window)
    q_hat_s = _local_mean(q_hat, half)
    qd_hat = _local_linear_slope(q_hat, config.timestep, half)
    qdd_hat = _local_linear_slope(qd_hat, config.timestep, max(5, half // 2))

    # Gate residual: visual proprio + plant qacc (declared in report).
    r_v = _visual_residual_stream(
        model, data, q=q_hat_s, qd=qd_hat, qacc=qacc, tau=tau_a, nominal=nominal
    )
    # Diagnostic: fully visual kinematics including FD acceleration.
    r_v_full = _visual_residual_stream(
        model, data, q=q_hat_s, qd=qd_hat, qacc=qdd_hat, tau=tau_a, nominal=nominal
    )

    q_err = _wrap_angle(q_hat_s - q_o)
    q_rmse = float(np.sqrt(np.mean(np.square(q_err))))
    # Also report continuous-track error after flip-aware unwrap.
    q_err_unwrapped = q_hat_s - q_o
    q_rmse_unwrapped = float(np.sqrt(np.mean(np.square(q_err_unwrapped))))
    qd_rmse = float(np.sqrt(np.mean(np.square(qd_hat - qd_o))))
    nrmse_o = float(residual_nrmse(r_o, tau_a))
    nrmse_v = float(residual_nrmse(r_v, tau_a))
    nrmse_v_full = float(residual_nrmse(r_v_full, tau_a))
    r_pseudo = r_v - r_o
    pseudo_rmse = float(np.sqrt(np.mean(np.square(r_pseudo))))

    g_phys = bool(nrmse_o < config.nrmse_oracle_max)
    g_geom = bool(q_rmse < config.q_rmse_max)
    g_res = bool(nrmse_v < config.nrmse_visual_max)
    return {
        "seed": seed,
        "trajectory": trajectory,
        "host_plant_id": VIS_HOST_PLANT_ID,
        "xml_sha256": simx_hinge_vis_xml_sha256(config.timestep),
        "plant_inventory": inventory,
        "n_steps": n_steps,
        "timestep": config.timestep,
        "q_rmse": q_rmse,
        "q_rmse_unwrapped": q_rmse_unwrapped,
        "qd_rmse": qd_rmse,
        "nrmse_oracle": nrmse_o,
        "nrmse_visual": nrmse_v,
        "nrmse_visual_full_fd_acc": nrmse_v_full,
        "pseudo_residual_rmse": pseudo_rmse,
        "min_panel_pixels": int(min(n_panel)) if n_panel else 0,
        "g_phys": g_phys,
        "g_geom": g_geom,
        "g_res": g_res,
        "cell_pass": bool(g_phys and g_geom and g_res),
        "perception": {
            "inputs": ["seg_gt", "depth"],
            "method": "pivot_centered_pca_xy",
            "qdot_method": "local_linear_slope_on_unwrapped_qhat",
            "drive": "pd_tracks_bounded_reference_oracle_state_for_u_only",
            "residual_visual_uses": "qhat_qdhat_plant_qacc",
            "forbidden": ["rgb_only", "learned_detector", "domain_randomization"],
        },
        "arrays": {
            "q_oracle": q_o,
            "qd_oracle": qd_o,
            "qacc_plant": qacc,
            "q_hat_raw": np.asarray(q_hat_raw, dtype=np.float64),
            "q_hat": q_hat_s,
            "qd_hat": qd_hat,
            "tau": tau_a,
            "residual_oracle": r_o,
            "residual_visual": r_v,
            "residual_visual_full_fd_acc": r_v_full,
            "residual_pseudo": r_pseudo,
            "n_panel_pixels": np.asarray(n_panel, dtype=np.int32),
        },
    }


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "vis-x0"
        handle.attrs["host_plant_id"] = VIS_HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["real_perception"] = False
        handle.attrs["unlocks_r10_c0"] = False
        meta_grp = handle.create_group("metadata")
        meta_grp.attrs["json"] = json.dumps(meta, sort_keys=True, default=str)
        learner = handle.create_group("learner_visible")
        for key in (
            "q_hat",
            "qd_hat",
            "tau",
            "residual_visual",
            "residual_pseudo",
        ):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_truth_qfrc_passive"] = True
        raw = handle.create_group("raw_truth")
        for key in (
            "q_oracle",
            "qd_oracle",
            "qacc_plant",
            "q_hat_raw",
            "residual_oracle",
            "residual_visual_full_fd_acc",
            "n_panel_pixels",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        raw.attrs["audit_only"] = True
        evaluation = handle.create_group("evaluation")
        evaluation.attrs["q_rmse"] = cell["q_rmse"]
        evaluation.attrs["nrmse_oracle"] = cell["nrmse_oracle"]
        evaluation.attrs["nrmse_visual"] = cell["nrmse_visual"]
        evaluation.attrs["cell_pass"] = cell["cell_pass"]


def run_vis_x0(
    output: str | Path,
    *,
    config: VISX0Config | None = None,
) -> dict[str, Any]:
    cfg = config or VISX0Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("VIS-X0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)
    (root / "simx_hinge_vis.xml").write_text(
        simx_hinge_vis_xml(cfg.timestep), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for trajectory in cfg.trajectories:
            cell = _rollout_cell(cfg, seed=seed, trajectory=trajectory)
            rel = f"seed_{seed}/{trajectory}.h5"
            _write_cell_h5(root / rel, cell)
            rows.append(
                {
                    "seed": seed,
                    "trajectory": trajectory,
                    "path": str(root / rel),
                    "q_rmse": cell["q_rmse"],
                    "q_rmse_unwrapped": cell["q_rmse_unwrapped"],
                    "qd_rmse": cell["qd_rmse"],
                    "nrmse_oracle": cell["nrmse_oracle"],
                    "nrmse_visual": cell["nrmse_visual"],
                    "nrmse_visual_full_fd_acc": cell["nrmse_visual_full_fd_acc"],
                    "pseudo_residual_rmse": cell["pseudo_residual_rmse"],
                    "min_panel_pixels": cell["min_panel_pixels"],
                    "g_phys": cell["g_phys"],
                    "g_geom": cell["g_geom"],
                    "g_res": cell["g_res"],
                    "cell_pass": cell["cell_pass"],
                }
            )

    g_phys = all(bool(r["g_phys"]) for r in rows)
    g_geom = all(bool(r["g_geom"]) for r in rows)
    g_res = all(bool(r["g_res"]) for r in rows)
    g_label = True
    passed = bool(g_phys and g_geom and g_res and g_label and len(rows) == 9)
    mujoco = import_mujoco_with_renderer()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "VIS-X0",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": VIS_HOST_PLANT_ID,
        "dynamics_sibling_of": "simx_hinge.v1",
        "xml_sha256": simx_hinge_vis_xml_sha256(cfg.timestep),
        "prereg": PREREG_PATH,
        "scientific_claim": "perception-interface smoke on SIM-X hinge; not real vision",
        "real_physics": False,
        "real_perception": False,
        "unlocks_r10_c0": False,
        "unlocks_vis_x1": passed,
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "floors": {
            "q_rmse_max": cfg.q_rmse_max,
            "nrmse_oracle_max": cfg.nrmse_oracle_max,
            "nrmse_visual_max": cfg.nrmse_visual_max,
            "note": "visual residual uses qhat, qdhat, plant qacc",
        },
        "cells": rows,
        "cell_count": len(rows),
        "max_q_rmse": max((float(r["q_rmse"]) for r in rows), default=None),
        "max_nrmse_oracle": max((float(r["nrmse_oracle"]) for r in rows), default=None),
        "max_nrmse_visual": max((float(r["nrmse_visual"]) for r in rows), default=None),
        "g_phys": g_phys,
        "g_geom": g_geom,
        "g_res": g_res,
        "g_label": g_label,
        "vis_x0_passed": passed,
        "pass_iff": (
            "oracle accounting <1e-4 AND q_rmse <0.05 AND "
            "visual(q,qd)+plant_qacc nrmse <0.10 on all 9 cells"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
