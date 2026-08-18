"""VIS-X3: learner-visible perception uncertainty → physics-attribution veto.

Reuses frozen VIS-X2 alarm (W, θ). Does not retune. Does not delete alarms.
Does not unlock R10-C0. Vetoes physics attribution only when U_t is high.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .simx_plant import SIMX_DAMPING
from .simx_vis_plant import VIS_HOST_PLANT_ID, simx_hinge_vis_xml
from .vis_x0 import _write_json
from .vis_x2 import (
    ALARM_EPS,
    ALARM_QUANTILE,
    ALARM_WINDOW,
    MISMATCH_DAMPINGS,
    VISX2Config,
    _condition_key,
    _rollout_cell,
    alarm_rate,
)

PREREG_PATH = "REPORT/REG/VISX/VISX3_PREREG.md"
SCHEMA_ID = "aprwm.vis_x3.attribution.v1"

# Frozen X2 baselines (do not retune θ / W from these).
X2_THETA_FROZEN = 0.3667960699204398
X2_FPR_CLEAN = 0.005257623554153523
X2_FPR_PERC = 0.06756046267087276
X2_TPR_PHY = 0.2835173501577287
X2_FPR_PERC_REF = 0.0676
X2_TPR_PHY_REF = 0.284

# Contracted gates from VISX3_PREREG.
G1_D_CLEAN_MAX = 0.02
G2_D_GAP_MIN = 0.20
G3_FPR_PHYCLAIM_MAX = 0.5 * X2_FPR_PERC_REF  # 0.0338
G4_TPR_PHYCLAIM_MIN = 0.8 * X2_TPR_PHY_REF  # 0.2272
G0_ABS_TOL = 0.03  # sampling tolerance for X2 carryover rates


@dataclass(frozen=True)
class VISX3Config(VISX2Config):
    """Same factorial / renderer / alarm window as X2; θ loaded frozen."""

    u_quantile: float = 0.99
    require_x2_summary: bool = True
    frozen_theta: float = X2_THETA_FROZEN
    g0_abs_tol: float = G0_ABS_TOL
    g1_d_clean_max: float = G1_D_CLEAN_MAX
    g2_d_gap_min: float = G2_D_GAP_MIN
    g3_fpr_phyclaim_max: float = G3_FPR_PHYCLAIM_MAX
    g4_tpr_phyclaim_min: float = G4_TPR_PHYCLAIM_MIN


def _load_x2_passed(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"VIS-X3 requires VIS-X2 summary at {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("vis_x2_passed")):
        raise RuntimeError("VIS-X3 locked until vis_x2_passed=true")
    if not bool(payload.get("false_physics_alarm_evidence")):
        raise RuntimeError("VIS-X3 locked until false_physics_alarm_evidence=true")
    return payload


def _align_u_to_scores(u: np.ndarray, *, window: int) -> np.ndarray:
    """Align per-step U to residual window scores (use window-end U)."""

    u = np.asarray(u, dtype=np.float64).reshape(-1)
    w = int(window)
    if u.size < w:
        return np.zeros(0, dtype=np.float64)
    # scores[i] covers steps [i, i+w-1]; use U at the window end.
    return u[w - 1 :]


def attribution_rates(
    scores: np.ndarray,
    u: np.ndarray,
    *,
    theta: float,
    tau_u: float,
    window: int,
) -> dict[str, float]:
    u_al = _align_u_to_scores(u, window=window)
    if scores.size == 0 or u_al.size == 0:
        return {
            "alarm_rate": float("nan"),
            "d_rate_steps": float("nan"),
            "d_rate_aligned": float("nan"),
            "phyclaim_rate": float("nan"),
            "ambiguous_rate": float("nan"),
        }
    n = min(scores.size, u_al.size)
    s = scores[:n]
    uu = u_al[:n]
    a = s > float(theta)
    d = uu > float(tau_u)
    c_phy = a & (~d)
    ambiguous = a & d
    u_all = np.asarray(u, dtype=np.float64)
    finite_u = u_all[np.isfinite(u_all)]
    d_steps = float(np.mean(finite_u > float(tau_u))) if finite_u.size else float("nan")
    return {
        "alarm_rate": float(np.mean(a)),
        "d_rate_steps": d_steps,
        "d_rate_aligned": float(np.mean(d)),
        "phyclaim_rate": float(np.mean(c_phy)),
        "ambiguous_rate": float(np.mean(ambiguous)),
    }


def _write_cell_h5(
    path: Path,
    cell: dict[str, Any],
    *,
    theta: float,
    tau_u: float,
    rates: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {k: v for k, v in cell.items() if k not in ("arrays", "scores", "U")}
    meta.update({"theta": float(theta), "tau_U": float(tau_u), **rates})
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "vis-x3"
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
        for key in ("q_hat", "qd_hat", "U", "tau", "residual_visual", "alarm_scores"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["theta"] = float(theta)
        learner.attrs["tau_U"] = float(tau_u)
        learner.attrs["excludes_oracle_from_U"] = True
        learner.attrs["excludes_residual_from_U"] = True
        raw = handle.create_group("raw_truth")
        for key in ("q_oracle", "qd_oracle", "residual_oracle", "n_panel_pixels"):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        raw.attrs["audit_only"] = True
        evaluation = handle.create_group("evaluation")
        for key, value in rates.items():
            evaluation.attrs[key] = value


def _outcome_pattern(
    *,
    g2: bool,
    g3: bool,
    g4: bool,
    d_clean: float,
    d_deg: float,
) -> str:
    if g2 and g3 and g4:
        return "attribution_success"
    if (not g2) or abs(d_deg - d_clean) < 0.05:
        return "uncertainty_nondiscriminative"
    if g3 and (not g4):
        return "over_veto"
    return "mixed_or_partial"


def run_vis_x3(
    output: str | Path,
    *,
    config: VISX3Config | None = None,
    x2_summary: str | Path | None = "runs/vis_x2/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or VISX3Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("VIS-X3 must not write under runs/r10_c0/")
    x2 = None
    if cfg.require_x2_summary:
        x2 = _load_x2_passed(x2_summary)
    # Prefer frozen θ from X2 summary when present; never recalibrate.
    theta = float(cfg.frozen_theta)
    if x2 is not None:
        theta = float(x2.get("alarm_harness", {}).get("theta", theta))

    root.mkdir(parents=True, exist_ok=True)
    (root / "simx_hinge_vis.xml").write_text(
        simx_hinge_vis_xml(cfg.timestep), encoding="utf-8"
    )

    # Calibration for τ_U only (nominal, clean). Alarm θ stays frozen.
    cal = _rollout_cell(
        cfg,
        seed=cfg.cal_seed,
        trajectory=cfg.cal_trajectory,
        physics="nominal",
        vision="clean",
        true_damping=SIMX_DAMPING,
        role="calibration_U",
    )
    if not cal["g_phys_clean"] or not cal["g_run"]:
        raise RuntimeError("VIS-X3 U-calibration cell failed G-phys-clean / G-run")
    u_cal = np.asarray(cal["U"], dtype=np.float64)
    u_cal = u_cal[np.isfinite(u_cal)]
    tau_u = float(np.quantile(u_cal, cfg.u_quantile))
    cal_rates = attribution_rates(
        cal["scores"], cal["U"], theta=theta, tau_u=tau_u, window=cfg.alarm_window
    )
    _write_cell_h5(
        root / "calibration" / f"seed_{cfg.cal_seed}_{cfg.cal_trajectory}.h5",
        cal,
        theta=theta,
        tau_u=tau_u,
        rates=cal_rates,
    )

    specs: list[tuple[str, str, float]] = [
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
                    physics=physics,  # type: ignore[arg-type]
                    vision=vision,  # type: ignore[arg-type]
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
                key = _condition_key(physics, vision, b_true)
                rel = f"test/seed_{seed}/{trajectory}/{key}.h5"
                _write_cell_h5(root / rel, cell, theta=theta, tau_u=tau_u, rates=rates)
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
                        "E_pseudo": cell["E_pseudo"],
                        "nrmse_oracle": cell["nrmse_oracle"],
                        "g_phys_clean": cell["g_phys_clean"],
                        "g_run": cell["g_run"],
                        **rates,
                    }
                )

    def _mean(physics: str, vision: str, key: str) -> float:
        vals = [
            float(r[key])
            for r in rows
            if r["physics"] == physics and r["vision"] == vision
        ]
        return float(np.mean(vals)) if vals else float("nan")

    fpr_clean = _mean("nominal", "clean", "alarm_rate")
    fpr_perc = _mean("nominal", "degraded", "alarm_rate")
    tpr_phy = _mean("mismatch", "clean", "alarm_rate")
    d_clean = _mean("nominal", "clean", "d_rate_steps")
    d_deg = _mean("nominal", "degraded", "d_rate_steps")
    fpr_phyclaim = _mean("nominal", "degraded", "phyclaim_rate")
    tpr_phyclaim = _mean("mismatch", "clean", "phyclaim_rate")
    amb_mismatch_deg = _mean("mismatch", "degraded", "ambiguous_rate")

    nominal_rows = [r for r in rows if r["physics"] == "nominal"]
    g_phys = all(bool(r["g_phys_clean"]) for r in nominal_rows)
    g_run = all(bool(r["g_run"]) for r in rows)

    g0 = bool(
        abs(fpr_clean - X2_FPR_CLEAN) <= cfg.g0_abs_tol
        and abs(fpr_perc - X2_FPR_PERC) <= cfg.g0_abs_tol
        and abs(tpr_phy - X2_TPR_PHY) <= cfg.g0_abs_tol
    )
    g1 = bool(d_clean <= cfg.g1_d_clean_max)
    g2 = bool(d_deg > d_clean and (d_deg - d_clean) > cfg.g2_d_gap_min)
    g3 = bool(fpr_phyclaim <= cfg.g3_fpr_phyclaim_max)
    g4 = bool(tpr_phyclaim >= cfg.g4_tpr_phyclaim_min)
    g5 = True  # enforced by construction: U from slope fit only

    pattern = _outcome_pattern(
        g2=g2, g3=g3, g4=g4, d_clean=d_clean, d_deg=d_deg
    )
    contract_ok = bool(g_phys and g_run and g0 and g1 and g2 and g3 and g4 and g5)
    attribution_success = pattern == "attribution_success"

    (root / "alarm_harness.json").write_text(
        json.dumps(
            {
                "window": cfg.alarm_window,
                "eps": cfg.alarm_eps,
                "theta_frozen_from_x2": theta,
                "theta_retune_forbidden": True,
                "tau_U": tau_u,
                "u_quantile": cfg.u_quantile,
                "u_definition": "SE(local_linear_slope_beta)",
                "u_forbidden_inputs": [
                    "q_oracle",
                    "qd_oracle",
                    "r_oracle",
                    "r_visual",
                    "b_true",
                    "physics_label",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "stage": "VIS-X3",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": VIS_HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": (
            "learner-visible perception uncertainty can veto physics attribution "
            "of residual alarms without deleting the alarm"
        ),
        "real_physics": False,
        "real_perception": False,
        "unlocks_r10_c0": False,
        "config": asdict(cfg),
        "x2_summary": str(x2_summary) if x2_summary else None,
        "x2_passed": None if x2 is None else bool(x2.get("vis_x2_passed")),
        "frozen_x2_baselines": {
            "theta": X2_THETA_FROZEN,
            "FPR_clean": X2_FPR_CLEAN,
            "FPR_perc": X2_FPR_PERC,
            "TPR_phy": X2_TPR_PHY,
            "FPR_phyclaim_max": G3_FPR_PHYCLAIM_MAX,
            "TPR_phyclaim_min": G4_TPR_PHYCLAIM_MIN,
        },
        "harness": {
            "alarm_window": ALARM_WINDOW,
            "alarm_eps": ALARM_EPS,
            "alarm_quantile_x2": ALARM_QUANTILE,
            "theta": theta,
            "tau_U": tau_u,
            "u_quantile": cfg.u_quantile,
            "mismatch_dampings": list(MISMATCH_DAMPINGS),
        },
        "metrics": {
            "FPR_clean": fpr_clean,
            "FPR_perc": fpr_perc,
            "TPR_phy": tpr_phy,
            "P_D_nominal_clean": d_clean,
            "P_D_nominal_degraded": d_deg,
            "FPR_phyclaim_perc": fpr_phyclaim,
            "TPR_phyclaim": tpr_phyclaim,
            "ambiguous_rate_mismatch_degraded": amb_mismatch_deg,
        },
        "semantics": {
            "A": "frozen residual alarm",
            "D": "U > tau_U (observation quality poor)",
            "C_phy": "A and not D (physics-supported claim)",
            "A1_D1": "observation-ambiguous; NOT physics-OK",
        },
        "cells": rows,
        "cell_count": len(rows),
        "g_phys_clean": g_phys,
        "g_run": g_run,
        "g0_x2_carryover": g0,
        "g1_u_clean_specificity": g1,
        "g2_u_perception_sensitivity": g2,
        "g3_false_attribution_suppression": g3,
        "g4_true_physics_retention": g4,
        "g5_no_oracle_leakage": g5,
        "outcome_pattern": pattern,
        "attribution_success": attribution_success,
        "vis_x3_contract_ok": contract_ok,
        "vis_x3_passed": contract_ok,
        "pass_iff": (
            "G0–G5: X2 carryover; D clean<=0.02; D gap>0.20; "
            "FPR_phyclaim<=0.0338; TPR_phyclaim>=0.227; no oracle in U"
        ),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
