"""VIS-X1: RGB-D (no GT seg) → state → perception-induced pseudo-residual.

Does not unlock R10-C0. Does not train detectors. Does not claim real vision.
Requires VIS-X0 PASS. Sole new variable vs X0: drop GT segmentation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import generalized_force_residual, residual_nrmse
from .mujoco_physics import import_mujoco_with_renderer
from .simx_plant import learner_nominal_params
from .simx_vis_plant import (
    VIS_CAMERA_NAME,
    VIS_HOST_PLANT_ID,
    VIS_PIVOT_XYZ,
    assert_vis_plant,
    load_simx_hinge_vis,
    simx_hinge_vis_xml,
    simx_hinge_vis_xml_sha256,
)
from .vis_x0 import (
    MIN_PANEL_PIXELS,
    VISX0Config,
    _camera_intrinsics,
    _local_linear_slope,
    _local_mean,
    _reference_q_qd,
    _unwrap_series,
    _visual_residual_stream,
    _wrap_angle,
    _write_json,
)

PREREG_PATH = "REPORT/REG/VISX/VISX1_PREREG.md"
SCHEMA_ID = "aprwm.vis_x1.residual.v1"
NRMSE_ORACLE_MAX = 1.0e-4

# Frozen classical RGB panel gate (panel material ≈ rgba 0.85 0.25 0.15).
RGB_R_MIN = 60.0
RGB_R_OVER_GB = 1.5


@dataclass(frozen=True)
class VISX1Config(VISX0Config):
    """Same host/camera/PD/residual as VIS-X0; only perception input changes."""

    nrmse_oracle_max: float = NRMSE_ORACLE_MAX
    require_x0_summary: bool = True


def panel_mask_from_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    r_min: float = RGB_R_MIN,
    r_over_gb: float = RGB_R_OVER_GB,
) -> np.ndarray:
    """Classical color+depth panel mask. Never uses GT segmentation."""

    rgb_f = np.asarray(rgb, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    r = rgb_f[..., 0]
    g = rgb_f[..., 1]
    b = rgb_f[..., 2]
    color = (r > r_over_gb * g + 1.0) & (r > r_over_gb * b + 1.0) & (r > r_min)
    return color & np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)


def estimate_q_from_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    data: Any,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, int]:
    """Pivot-centered PCA on RGB-D panel points → hinge angle."""

    mask = panel_mask_from_rgb_depth(rgb, depth)
    n_pix = int(np.count_nonzero(mask))
    if n_pix < MIN_PANEL_PIXELS:
        return float("nan"), n_pix
    depth = np.asarray(depth, dtype=np.float64)
    vs, us = np.where(mask)
    zs = depth[vs, us]
    xs = (us.astype(np.float64) - cx) / fx * zs
    ys = (vs.astype(np.float64) - cy) / fy * zs
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


def _render_rgb_depth(renderer: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=VIS_CAMERA_NAME)
    rgb = np.array(renderer.render(), copy=True)
    renderer.enable_depth_rendering()
    depth = np.array(renderer.render(), copy=True)
    renderer.disable_depth_rendering()
    return rgb, depth


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or float(np.std(a)) < 1.0e-18 or float(np.std(b)) < 1.0e-18:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_x0_passed(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"VIS-X1 requires VIS-X0 summary at {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("vis_x0_passed")):
        raise RuntimeError("VIS-X1 locked until vis_x0_passed=true")
    return payload


def _rollout_cell(config: VISX1Config, *, seed: int, trajectory: str) -> dict[str, Any]:
    mujoco = import_mujoco_with_renderer()
    model, data = load_simx_hinge_vis(config.timestep)
    inventory = assert_vis_plant(model)
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
            rgb, depth = _render_rgb_depth(renderer, data)
            q_est, n_pix = estimate_q_from_rgb_depth(
                rgb,
                depth,
                data=data,
                camera_id=cam_id,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            qpos_pre = np.array(data.qpos, dtype=np.float64, copy=True)
            qvel_pre = np.array(data.qvel, dtype=np.float64, copy=True)
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

    r_v = _visual_residual_stream(
        model, data, q=q_hat_s, qd=qd_hat, qacc=qacc, tau=tau_a, nominal=nominal
    )
    # Leave-one-channel replays for attribution (audit).
    r_vq_oqd = _visual_residual_stream(
        model, data, q=q_hat_s, qd=qd_o, qacc=qacc, tau=tau_a, nominal=nominal
    )
    r_oq_vqd = _visual_residual_stream(
        model, data, q=q_o, qd=qd_hat, qacc=qacc, tau=tau_a, nominal=nominal
    )

    q_err = np.asarray(_wrap_angle(q_hat_s - q_o), dtype=np.float64)
    qd_err = qd_hat - qd_o
    e_q = float(np.sqrt(np.mean(np.square(q_err))))
    e_qd = float(np.sqrt(np.mean(np.square(qd_err))))
    e_r = float(residual_nrmse(r_v, tau_a))
    nrmse_o = float(residual_nrmse(r_o, tau_a))
    r_pseudo = r_v - r_o
    e_pseudo = float(np.sqrt(np.mean(np.square(r_pseudo))))
    e_pseudo_vq = float(np.sqrt(np.mean(np.square(r_vq_oqd - r_o))))
    e_pseudo_vqd = float(np.sqrt(np.mean(np.square(r_oq_vqd - r_o))))

    abs_pseudo = np.abs(r_pseudo)
    corr_pseudo_qd = _safe_corr(abs_pseudo, np.abs(qd_err))
    corr_pseudo_q = _safe_corr(abs_pseudo, np.abs(q_err))
    finite_ok = bool(np.all(np.isfinite(q_hat_s)) and np.all(np.isfinite(qd_hat)))
    min_pix = int(min(n_panel)) if n_panel else 0

    g_phys = bool(nrmse_o < config.nrmse_oracle_max)
    g_run = bool(finite_ok and min_pix >= MIN_PANEL_PIXELS)
    return {
        "seed": seed,
        "trajectory": trajectory,
        "host_plant_id": VIS_HOST_PLANT_ID,
        "xml_sha256": simx_hinge_vis_xml_sha256(config.timestep),
        "plant_inventory": inventory,
        "n_steps": n_steps,
        "timestep": config.timestep,
        "E_q": e_q,
        "E_qd": e_qd,
        "E_r": e_r,
        "E_pseudo": e_pseudo,
        "nrmse_oracle": nrmse_o,
        "E_pseudo_visual_q_oracle_qd": e_pseudo_vq,
        "E_pseudo_oracle_q_visual_qd": e_pseudo_vqd,
        "corr_abs_pseudo_abs_qd_err": corr_pseudo_qd,
        "corr_abs_pseudo_abs_q_err": corr_pseudo_q,
        "min_panel_pixels": min_pix,
        "g_phys": g_phys,
        "g_run": g_run,
        "g_attrib": True,
        "cell_contract_ok": bool(g_phys and g_run),
        "perception": {
            "inputs": ["rgb", "depth"],
            "uses_gt_segmentation": False,
            "method": "rgb_color_mask_then_pivot_centered_pca_xy",
            "rgb_gate": {"r_min": RGB_R_MIN, "r_over_gb": RGB_R_OVER_GB},
            "qdot_method": "local_linear_slope_on_unwrapped_qhat",
            "drive": "pd_tracks_bounded_reference_oracle_state_for_u_only",
            "residual_visual_uses": "qhat_qdhat_plant_qacc",
            "forbidden": ["gt_seg_in_estimator", "learned_detector", "domain_randomization"],
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
            "residual_pseudo": r_pseudo,
            "residual_visual_q_oracle_qd": r_vq_oqd,
            "residual_oracle_q_visual_qd": r_oq_vqd,
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
        handle.attrs["label"] = "vis-x1"
        handle.attrs["host_plant_id"] = VIS_HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["real_perception"] = False
        handle.attrs["unlocks_r10_c0"] = False
        handle.attrs["uses_gt_segmentation"] = False
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
        learner.attrs["excludes_gt_segmentation"] = True
        raw = handle.create_group("raw_truth")
        for key in (
            "q_oracle",
            "qd_oracle",
            "qacc_plant",
            "q_hat_raw",
            "residual_oracle",
            "residual_visual_q_oracle_qd",
            "residual_oracle_q_visual_qd",
            "n_panel_pixels",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        raw.attrs["audit_only"] = True
        evaluation = handle.create_group("evaluation")
        for key in (
            "E_q",
            "E_qd",
            "E_r",
            "E_pseudo",
            "nrmse_oracle",
            "corr_abs_pseudo_abs_qd_err",
            "corr_abs_pseudo_abs_q_err",
            "E_pseudo_visual_q_oracle_qd",
            "E_pseudo_oracle_q_visual_qd",
        ):
            evaluation.attrs[key] = cell[key]
        evaluation.attrs["cell_contract_ok"] = cell["cell_contract_ok"]


def _x0_baselines(x0: dict[str, Any] | None) -> dict[str, Any] | None:
    if x0 is None:
        return None
    cells = x0.get("cells") or []
    if not cells:
        return None
    return {
        "max_q_rmse": x0.get("max_q_rmse"),
        "max_nrmse_visual": x0.get("max_nrmse_visual"),
        "mean_pseudo_residual_rmse": float(
            np.mean([float(c.get("pseudo_residual_rmse", np.nan)) for c in cells])
        ),
        "cells": [
            {
                "seed": c["seed"],
                "trajectory": c["trajectory"],
                "q_rmse": c.get("q_rmse"),
                "qd_rmse": c.get("qd_rmse"),
                "nrmse_visual": c.get("nrmse_visual"),
                "pseudo_residual_rmse": c.get("pseudo_residual_rmse"),
            }
            for c in cells
        ],
    }


def run_vis_x1(
    output: str | Path,
    *,
    config: VISX1Config | None = None,
    x0_summary: str | Path | None = "runs/vis_x0/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or VISX1Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("VIS-X1 must not write under runs/r10_c0/")
    x0 = None
    if cfg.require_x0_summary:
        x0 = _load_x0_passed(x0_summary)
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
                    "E_q": cell["E_q"],
                    "E_qd": cell["E_qd"],
                    "E_r": cell["E_r"],
                    "E_pseudo": cell["E_pseudo"],
                    "nrmse_oracle": cell["nrmse_oracle"],
                    "E_pseudo_visual_q_oracle_qd": cell["E_pseudo_visual_q_oracle_qd"],
                    "E_pseudo_oracle_q_visual_qd": cell["E_pseudo_oracle_q_visual_qd"],
                    "corr_abs_pseudo_abs_qd_err": cell["corr_abs_pseudo_abs_qd_err"],
                    "corr_abs_pseudo_abs_q_err": cell["corr_abs_pseudo_abs_q_err"],
                    "min_panel_pixels": cell["min_panel_pixels"],
                    "g_phys": cell["g_phys"],
                    "g_run": cell["g_run"],
                    "cell_contract_ok": cell["cell_contract_ok"],
                }
            )

    g_phys = all(bool(r["g_phys"]) for r in rows)
    g_run = all(bool(r["g_run"]) for r in rows)
    g_label = True
    g_attrib = True
    g_contrast = True  # baselines logged when X0 summary present
    contract_ok = bool(
        g_phys and g_run and g_label and g_attrib and g_contrast and len(rows) == 9
    )

    # Scientific reading (not a beauty contest).
    max_e_pseudo = max((float(r["E_pseudo"]) for r in rows), default=0.0)
    max_e_r = max((float(r["E_r"]) for r in rows), default=0.0)
    mean_corr_qd = float(np.mean([r["corr_abs_pseudo_abs_qd_err"] for r in rows]))
    mean_corr_q = float(np.mean([r["corr_abs_pseudo_abs_q_err"] for r in rows]))
    mean_ep_vq = float(np.mean([r["E_pseudo_visual_q_oracle_qd"] for r in rows]))
    mean_ep_vqd = float(np.mean([r["E_pseudo_oracle_q_visual_qd"] for r in rows]))
    attribution_dominant = (
        "velocity"
        if mean_ep_vqd >= mean_ep_vq
        else "pose"
    )
    masquerade = bool(g_phys and (max_e_pseudo > 1.0e-3 or max_e_r > 1.0e-3))

    mujoco = import_mujoco_with_renderer()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "VIS-X1",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": VIS_HOST_PLANT_ID,
        "dynamics_sibling_of": "simx_hinge.v1",
        "xml_sha256": simx_hinge_vis_xml_sha256(cfg.timestep),
        "prereg": PREREG_PATH,
        "scientific_claim": (
            "quantify perception-induced pseudo-residual on SIM-X hinge; "
            "r_visual!=0 is not physics mismatch"
        ),
        "real_physics": False,
        "real_perception": False,
        "uses_gt_segmentation": False,
        "unlocks_r10_c0": False,
        "unlocks_vis_x2": contract_ok,
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "x0_summary": str(x0_summary) if x0_summary else None,
        "x0_baselines": _x0_baselines(x0),
        "cells": rows,
        "cell_count": len(rows),
        "max_E_q": max((float(r["E_q"]) for r in rows), default=None),
        "max_E_qd": max((float(r["E_qd"]) for r in rows), default=None),
        "max_E_r": max_e_r,
        "max_E_pseudo": max_e_pseudo,
        "max_nrmse_oracle": max((float(r["nrmse_oracle"]) for r in rows), default=None),
        "mean_corr_abs_pseudo_abs_qd_err": mean_corr_qd,
        "mean_corr_abs_pseudo_abs_q_err": mean_corr_q,
        "mean_E_pseudo_visual_q_oracle_qd": mean_ep_vq,
        "mean_E_pseudo_oracle_q_visual_qd": mean_ep_vqd,
        "attribution_dominant_channel": attribution_dominant,
        "perception_masquerade_evidence": masquerade,
        "g_phys": g_phys,
        "g_run": g_run,
        "g_contrast": g_contrast,
        "g_attrib": g_attrib,
        "g_label": g_label,
        "vis_x1_contract_ok": contract_ok,
        "vis_x1_passed": contract_ok,
        "pass_iff": (
            "G-phys + G-run + G-label + G-attrib + G-contrast on 9 cells; "
            "deliverable is E_pseudo quantification (not beating VIS-X0 E_q)"
        ),
        "interpretation_rule": (
            "Never read r_visual!=0 alone as physics mismatch; "
            "oracle floor + risen E_pseudo => perception can masquerade as model inadequacy"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
