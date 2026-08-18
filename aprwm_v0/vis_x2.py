"""VIS-X2: perception degradation → false residual-based physics diagnosis.

Does not unlock R10-C0. Does not retune the alarm after formal results.
Does not add perception-uncertainty veto (VIS-X3). Reuses VIS-X1 RGB-D estimator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from .mujoco_force import generalized_force_residual, residual_nrmse
from .mujoco_physics import import_mujoco_with_renderer
from .simx_plant import SIMX_DAMPING, learner_nominal_params
from .simx_vis_plant import (
    VIS_HOST_PLANT_ID,
    assert_vis_plant,
    load_simx_hinge_vis,
    simx_hinge_vis_xml,
    simx_hinge_vis_xml_sha256,
)
from .vis_x0 import (
    MIN_PANEL_PIXELS,
    _camera_intrinsics,
    _local_linear_slope_and_se,
    _local_mean,
    _reference_q_qd,
    _unwrap_series,
    _visual_residual_stream,
    _wrap_angle,
    _write_json,
)
from .vis_x0 import VISX0Config
from .vis_x1 import estimate_q_from_rgb_depth, _render_rgb_depth

PREREG_PATH = "REPORT/REG/VISX/VISX2_PREREG.md"
SCHEMA_ID = "aprwm.vis_x2.alarm.v1"
NRMSE_ORACLE_MAX = 1.0e-4

# Frozen alarm harness (declared before formal; do not retune post-hoc).
ALARM_WINDOW = 50
ALARM_EPS = 1.0e-8
ALARM_QUANTILE = 0.99

# Frozen degraded-observation family (renderer only; no plant / camera-pose change).
DEGRADE_RGB_SCALE = 0.55
DEGRADE_RGB_BIAS = (20.0, -15.0, 35.0)
DEGRADE_DEPTH_NOISE_STD = 0.04
DEGRADE_DEPTH_DROPOUT = 0.12
DEGRADE_OCCLUSION_FRAC = 0.28

MISMATCH_DAMPINGS = (0.05, 0.15)
PhysicsKind = Literal["nominal", "mismatch"]
VisionKind = Literal["clean", "degraded"]


@dataclass(frozen=True)
class VISX2Config(VISX0Config):
    duration_s: float = 2.0
    image_height: int = 160
    image_width: int = 240
    nrmse_oracle_max: float = NRMSE_ORACLE_MAX
    alarm_window: int = ALARM_WINDOW
    alarm_eps: float = ALARM_EPS
    alarm_quantile: float = ALARM_QUANTILE
    cal_seed: int = 9101
    cal_trajectory: str = "sine"
    test_seeds: tuple[int, ...] = (9111, 9121)
    test_trajectories: tuple[str, ...] = ("sine", "chirp")
    mismatch_dampings: tuple[float, ...] = MISMATCH_DAMPINGS
    # Convention floors frozen in report; used for G-pos / evidence flags only.
    tpr_margin_over_fpr: float = 0.05
    fpr_perc_margin_over_clean: float = 0.05
    require_x1_summary: bool = True


def degrade_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen illumination/color + depth noise/dropout + occlusion."""

    out_rgb = np.asarray(rgb, dtype=np.float64) * float(DEGRADE_RGB_SCALE)
    out_rgb = out_rgb + np.asarray(DEGRADE_RGB_BIAS, dtype=np.float64)
    out_rgb = np.clip(out_rgb, 0.0, 255.0).astype(np.uint8)

    out_depth = np.asarray(depth, dtype=np.float64).copy()
    noise = rng.normal(0.0, DEGRADE_DEPTH_NOISE_STD, size=out_depth.shape)
    out_depth = out_depth + noise
    drop = rng.random(out_depth.shape) < DEGRADE_DEPTH_DROPOUT
    out_depth = np.where(drop, 1.0e3, out_depth)

    h, w = out_rgb.shape[:2]
    occ_h = max(1, int(round(h * DEGRADE_OCCLUSION_FRAC)))
    occ_w = max(1, int(round(w * DEGRADE_OCCLUSION_FRAC)))
    y0 = int(rng.integers(0, max(1, h - occ_h + 1)))
    x0 = int(rng.integers(0, max(1, w - occ_w + 1)))
    out_rgb[y0 : y0 + occ_h, x0 : x0 + occ_w] = 0
    out_depth[y0 : y0 + occ_h, x0 : x0 + occ_w] = 1.0e3
    return out_rgb, out_depth


def window_scores(
    residual: np.ndarray,
    tau: np.ndarray,
    *,
    window: int,
    eps: float,
) -> np.ndarray:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    u = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = int(r.size)
    w = int(window)
    if n < w:
        return np.zeros(0, dtype=np.float64)
    # Cumulative sums for O(n) RMS windows.
    r2 = np.concatenate([[0.0], np.cumsum(r * r)])
    u2 = np.concatenate([[0.0], np.cumsum(u * u)])
    scores = np.empty(n - w + 1, dtype=np.float64)
    for t in range(w - 1, n):
        i0 = t - w + 1
        num = np.sqrt((r2[t + 1] - r2[i0]) / w)
        den = np.sqrt((u2[t + 1] - u2[i0]) / w) + float(eps)
        scores[i0] = num / den
    return scores


def alarm_rate(scores: np.ndarray, theta: float) -> float:
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores > float(theta)))


def _load_x1_passed(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"VIS-X2 requires VIS-X1 summary at {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("vis_x1_passed")):
        raise RuntimeError("VIS-X2 locked until vis_x1_passed=true")
    return payload


def _rollout_cell(
    config: VISX2Config,
    *,
    seed: int,
    trajectory: str,
    physics: PhysicsKind,
    vision: VisionKind,
    true_damping: float,
    role: str,
) -> dict[str, Any]:
    mujoco = import_mujoco_with_renderer()
    model, data = load_simx_hinge_vis(config.timestep, true_damping=true_damping)
    inventory = assert_vis_plant(model, true_damping=true_damping)
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
    rng = np.random.default_rng(
        int(seed)
        + (0 if vision == "clean" else 901_001)
        + 17 * abs(int.from_bytes(trajectory.encode(), "little") % 10_000)
        + 101 * abs(int(round(true_damping * 1000.0)))
        + (0 if physics == "nominal" else 50_003)
    )

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
            if vision == "degraded":
                rgb, depth = degrade_rgb_depth(rgb, depth, rng=rng)
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
    qd_hat, u_hat = _local_linear_slope_and_se(q_hat, config.timestep, half)
    r_v = _visual_residual_stream(
        model, data, q=q_hat_s, qd=qd_hat, qacc=qacc, tau=tau_a, nominal=nominal
    )
    r_pseudo = r_v - r_o

    e_q = float(np.sqrt(np.mean(np.square(_wrap_angle(q_hat_s - q_o)))))
    e_qd = float(np.sqrt(np.mean(np.square(qd_hat - qd_o))))
    e_r = float(residual_nrmse(r_v, tau_a))
    e_pseudo = float(np.sqrt(np.mean(np.square(r_pseudo))))
    nrmse_o = float(residual_nrmse(r_o, tau_a))
    scores = window_scores(
        r_v, tau_a, window=config.alarm_window, eps=config.alarm_eps
    )
    finite_ok = bool(np.all(np.isfinite(q_hat_s)) and np.all(np.isfinite(qd_hat)))
    min_pix = int(min(n_panel)) if n_panel else 0
    # Perception-only cells must keep oracle accounting closed.
    if physics == "nominal":
        g_phys_clean = bool(nrmse_o < config.nrmse_oracle_max)
    else:
        g_phys_clean = True  # not required; mismatch is positive control
    g_run = bool(finite_ok and min_pix >= MIN_PANEL_PIXELS)

    return {
        "role": role,
        "seed": seed,
        "trajectory": trajectory,
        "physics": physics,
        "vision": vision,
        "true_damping": float(true_damping),
        "learner_damping": float(SIMX_DAMPING),
        "host_plant_id": VIS_HOST_PLANT_ID,
        "xml_sha256": simx_hinge_vis_xml_sha256(
            config.timestep, true_damping=true_damping
        ),
        "plant_inventory": inventory,
        "n_steps": n_steps,
        "E_q": e_q,
        "E_qd": e_qd,
        "E_r": e_r,
        "E_pseudo": e_pseudo,
        "nrmse_oracle": nrmse_o,
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
        },
    }


def _write_cell_h5(path: Path, cell: dict[str, Any], *, theta: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {k: v for k, v in cell.items() if k not in ("arrays", "scores", "U")}
    if theta is not None:
        meta["theta"] = float(theta)
        meta["alarm_rate"] = alarm_rate(cell["scores"], theta)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "vis-x2"
        handle.attrs["host_plant_id"] = VIS_HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["real_perception"] = False
        handle.attrs["unlocks_r10_c0"] = False
        handle.attrs["physics"] = cell["physics"]
        handle.attrs["vision"] = cell["vision"]
        handle.attrs["true_damping"] = cell["true_damping"]
        meta_grp = handle.create_group("metadata")
        meta_grp.attrs["json"] = json.dumps(meta, sort_keys=True, default=str)
        learner = handle.create_group("learner_visible")
        for key in ("q_hat", "qd_hat", "tau", "residual_visual", "residual_pseudo"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.create_dataset("alarm_scores", data=arrays["alarm_scores"], compression="gzip")
        if theta is not None:
            learner.attrs["theta"] = float(theta)
            learner.attrs["alarm_rate"] = alarm_rate(cell["scores"], theta)
        raw = handle.create_group("raw_truth")
        for key in (
            "q_oracle",
            "qd_oracle",
            "residual_oracle",
            "n_panel_pixels",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        raw.attrs["audit_only"] = True


def _condition_key(physics: str, vision: str, true_damping: float) -> str:
    if physics == "nominal":
        return f"nominal_{vision}"
    return f"mismatch_b{true_damping:g}_{vision}"


def run_vis_x2(
    output: str | Path,
    *,
    config: VISX2Config | None = None,
    x1_summary: str | Path | None = "runs/vis_x1/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or VISX2Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("VIS-X2 must not write under runs/r10_c0/")
    x1 = None
    if cfg.require_x1_summary:
        x1 = _load_x1_passed(x1_summary)
    root.mkdir(parents=True, exist_ok=True)
    (root / "simx_hinge_vis.xml").write_text(
        simx_hinge_vis_xml(cfg.timestep), encoding="utf-8"
    )
    (root / "alarm_harness.json").write_text(
        json.dumps(
            {
                "window": cfg.alarm_window,
                "eps": cfg.alarm_eps,
                "quantile": cfg.alarm_quantile,
                "note": "Frozen before formal; do not retune after degraded/mismatch",
                "degraded_family": {
                    "rgb_scale": DEGRADE_RGB_SCALE,
                    "rgb_bias": list(DEGRADE_RGB_BIAS),
                    "depth_noise_std": DEGRADE_DEPTH_NOISE_STD,
                    "depth_dropout": DEGRADE_DEPTH_DROPOUT,
                    "occlusion_frac": DEGRADE_OCCLUSION_FRAC,
                    "camera_pose_shift": False,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # --- Calibration: nominal + clean only ---
    cal = _rollout_cell(
        cfg,
        seed=cfg.cal_seed,
        trajectory=cfg.cal_trajectory,
        physics="nominal",
        vision="clean",
        true_damping=SIMX_DAMPING,
        role="calibration",
    )
    if not cal["g_phys_clean"] or not cal["g_run"]:
        raise RuntimeError("VIS-X2 calibration cell failed G-phys-clean / G-run")
    theta = float(np.quantile(cal["scores"], cfg.alarm_quantile))
    _write_cell_h5(root / "calibration" / f"seed_{cfg.cal_seed}_{cfg.cal_trajectory}.h5", cal, theta=theta)

    # --- Formal factorial ---
    specs: list[tuple[PhysicsKind, VisionKind, float]] = [
        ("nominal", "clean", SIMX_DAMPING),
        ("nominal", "degraded", SIMX_DAMPING),
    ]
    for b in cfg.mismatch_dampings:
        specs.append(("mismatch", "clean", float(b)))
        specs.append(("mismatch", "degraded", float(b)))

    rows: list[dict[str, Any]] = []
    for seed in cfg.test_seeds:
        for trajectory in cfg.test_trajectories:
            for physics, vision, b_true in specs:
                cell = _rollout_cell(
                    cfg,
                    seed=seed,
                    trajectory=trajectory,
                    physics=physics,
                    vision=vision,
                    true_damping=b_true,
                    role="test",
                )
                rate = alarm_rate(cell["scores"], theta)
                key = _condition_key(physics, vision, b_true)
                rel = f"test/seed_{seed}/{trajectory}/{key}.h5"
                _write_cell_h5(root / rel, cell, theta=theta)
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
                        "alarm_rate": rate,
                        "g_phys_clean": cell["g_phys_clean"],
                        "g_run": cell["g_run"],
                        "min_panel_pixels": cell["min_panel_pixels"],
                    }
                )

    def _mean_rate(physics: str, vision: str) -> float:
        vals = [
            float(r["alarm_rate"])
            for r in rows
            if r["physics"] == physics and r["vision"] == vision
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def _mean_metric(physics: str, vision: str, key: str) -> float:
        vals = [
            float(r[key])
            for r in rows
            if r["physics"] == physics and r["vision"] == vision
        ]
        return float(np.mean(vals)) if vals else float("nan")

    fpr_clean = _mean_rate("nominal", "clean")
    fpr_perc = _mean_rate("nominal", "degraded")
    tpr_phy = _mean_rate("mismatch", "clean")
    tpr_both = _mean_rate("mismatch", "degraded")

    e_qd_clean = _mean_metric("nominal", "clean", "E_qd")
    e_qd_deg = _mean_metric("nominal", "degraded", "E_qd")
    e_pseudo_clean = _mean_metric("nominal", "clean", "E_pseudo")
    e_pseudo_deg = _mean_metric("nominal", "degraded", "E_pseudo")

    nominal_rows = [r for r in rows if r["physics"] == "nominal"]
    g_phys_clean = all(bool(r["g_phys_clean"]) for r in nominal_rows)
    g_run = all(bool(r["g_run"]) for r in rows)
    g_harness = bool(np.isfinite(theta) and np.isfinite(fpr_clean))
    g_pos = bool(tpr_phy > fpr_clean + cfg.tpr_margin_over_fpr)
    g_perc = True  # reporting gate; evidence flag separate
    g_label = True

    false_alarm_evidence = bool(
        g_phys_clean
        and fpr_perc > fpr_clean + cfg.fpr_perc_margin_over_clean
    )
    chain_ok = bool(
        e_qd_deg > e_qd_clean
        and e_pseudo_deg > e_pseudo_clean
        and fpr_perc > fpr_clean
    )
    contract_ok = bool(g_phys_clean and g_run and g_harness and g_pos and g_label and g_perc)

    mujoco = import_mujoco_with_renderer()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "VIS-X2",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": VIS_HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": (
            "perception degradation can trigger false residual-based "
            "physics-invalidity alarms; r_visual!=0 is not physics mismatch"
        ),
        "real_physics": False,
        "real_perception": False,
        "unlocks_r10_c0": False,
        "unlocks_vis_x3": bool(contract_ok and false_alarm_evidence),
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "x1_summary": str(x1_summary) if x1_summary else None,
        "x1_passed": None if x1 is None else bool(x1.get("vis_x1_passed")),
        "alarm_harness": {
            "window": cfg.alarm_window,
            "eps": cfg.alarm_eps,
            "quantile": cfg.alarm_quantile,
            "theta": theta,
            "calibration": {
                "seed": cfg.cal_seed,
                "trajectory": cfg.cal_trajectory,
                "physics": "nominal",
                "vision": "clean",
                "n_scores": int(cal["scores"].size),
            },
            "retune_after_formal_forbidden": True,
            "perception_uncertainty_veto": False,
        },
        "metrics": {
            "FPR_clean": fpr_clean,
            "FPR_perc": fpr_perc,
            "TPR_phy": tpr_phy,
            "TPR_mismatch_degraded": tpr_both,
            "mean_E_qd_nominal_clean": e_qd_clean,
            "mean_E_qd_nominal_degraded": e_qd_deg,
            "mean_E_pseudo_nominal_clean": e_pseudo_clean,
            "mean_E_pseudo_nominal_degraded": e_pseudo_deg,
        },
        "attribution_chain": {
            "E_qd_rises": bool(e_qd_deg > e_qd_clean),
            "E_pseudo_rises": bool(e_pseudo_deg > e_pseudo_clean),
            "alarm_rate_rises": bool(fpr_perc > fpr_clean),
            "chain_holds": chain_ok,
        },
        "cells": rows,
        "cell_count": len(rows),
        "g_phys_clean": g_phys_clean,
        "g_run": g_run,
        "g_harness": g_harness,
        "g_pos": g_pos,
        "g_perc": g_perc,
        "g_label": g_label,
        "false_physics_alarm_evidence": false_alarm_evidence,
        "vis_x2_contract_ok": contract_ok,
        "vis_x2_passed": contract_ok,
        "pass_iff": (
            "G-phys-clean on nominal; harness θ from cal only; "
            "TPR_phy > FPR_clean + margin; report FPR_perc vs FPR_clean"
        ),
        "interpretation_rule": (
            "Never read r_visual alarm alone as physics mismatch when "
            "r_oracle≈0 on nominal+degraded"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
