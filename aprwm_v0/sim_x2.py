"""SIM-X2: action-conditioned validity information channel.

Analytic per-step MI under synthetic observation noise. Not X3
lifecycle; not R10; not hardware noise.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import generalized_force_residual, residual_nrmse
from .mujoco_physics import import_mujoco
from .simx_plant import (
    HOST_PLANT_ID,
    SIMX_DAMPING,
    assert_x1_plant,
    learner_nominal_params,
    load_simx_hinge,
)

PREREG_PATH = "REPORT/REG/SIMX/SIMX2_PREREG.md"
SCHEMA_ID = "aprwm.sim_x2.channel.v1"
EPS = 1.0e-8
ORACLE_TOL = 1.0e-4

SIMX2_SEEDS = (9101, 9111, 9121)
SIMX2_TRUE_DAMPINGS = (0.05, 0.10, 0.15)

# Primary + iso-energy. info ≡ low_f by design.
SIMX2_ACTIONS: dict[str, dict[str, float]] = {
    "cons": {"amplitude": 0.02, "freq_hz": 0.25},
    "info": {"amplitude": 0.10, "freq_hz": 0.25},
    "low_f": {"amplitude": 0.10, "freq_hz": 0.25},
    "high_f": {"amplitude": 0.10, "freq_hz": 5.0},
}


@dataclass(frozen=True)
class SIMX2Config:
    seeds: tuple[int, ...] = SIMX2_SEEDS
    true_dampings: tuple[float, ...] = SIMX2_TRUE_DAMPINGS
    learner_damping: float = SIMX_DAMPING
    duration_s: float = 10.0
    timestep: float = 0.002
    steady_t0_s: float = 2.0
    sigma_obs: float = 0.01
    oracle_tol: float = ORACLE_TOL
    i_cons_max: float = 0.05
    i_info_min: float = 0.25
    i_gap_min: float = 0.20
    applied_match_atol: float = 1.0e-12


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def action_torque_sequence(
    name: str,
    *,
    n_steps: int,
    dt: float,
    actions: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    spec = (actions or SIMX2_ACTIONS)[name]
    t = np.arange(n_steps, dtype=np.float64) * dt
    return float(spec["amplitude"]) * np.sin(2.0 * np.pi * float(spec["freq_hz"]) * t)


def _gauss_pdf(y: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    z = (y - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def mi_binary_v_given_mixture(
    *,
    mean_valid: float,
    mean_inv_a: float,
    mean_inv_b: float,
    sigma: float,
    n_grid: int = 2049,
) -> float:
    """I(V; Y) in bits for equal-prior binary V and Gaussian mixture invalid."""

    sigma = float(sigma)
    means = np.asarray([mean_valid, mean_inv_a, mean_inv_b], dtype=np.float64)
    lo = float(np.min(means) - 8.0 * sigma)
    hi = float(np.max(means) + 8.0 * sigma)
    y = np.linspace(lo, hi, n_grid, dtype=np.float64)
    dy = float(y[1] - y[0])
    p_v1 = _gauss_pdf(y, mean_valid, sigma)
    p_v0 = 0.5 * _gauss_pdf(y, mean_inv_a, sigma) + 0.5 * _gauss_pdf(
        y, mean_inv_b, sigma
    )
    p_y = 0.5 * p_v1 + 0.5 * p_v0
    # Avoid log(0); density support is covered by the grid.
    ratio_v1 = np.where(p_y > 0.0, p_v1 / np.maximum(p_y, 1.0e-300), 1.0)
    ratio_v0 = np.where(p_y > 0.0, p_v0 / np.maximum(p_y, 1.0e-300), 1.0)
    term = 0.5 * p_v1 * np.log2(np.maximum(ratio_v1, 1.0e-300)) + 0.5 * p_v0 * np.log2(
        np.maximum(ratio_v0, 1.0e-300)
    )
    return float(np.sum(term) * dy)


def _rollout_action_damping(
    *,
    torques: np.ndarray,
    timestep: float,
    true_damping: float,
    seed: int,
) -> dict[str, np.ndarray]:
    mujoco = import_mujoco()
    model, data = load_simx_hinge(timestep, true_damping=true_damping)
    assert_x1_plant(model, true_damping=true_damping)
    learner = learner_nominal_params()
    b0 = float(learner.damping)
    delta_b = float(true_damping) - b0

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.15 * np.sin(seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    qvel = np.zeros(torques.size, dtype=np.float64)
    residual = np.zeros(torques.size, dtype=np.float64)
    applied = np.zeros(torques.size, dtype=np.float64)
    r_oracle = np.zeros(torques.size, dtype=np.float64)

    for i, tau in enumerate(torques):
        qvel_pre = np.array(data.qvel, dtype=np.float64, copy=True)
        data.qfrc_applied[:] = 0.0
        data.qfrc_actuator[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        residual[i] = float(
            generalized_force_residual(
                model,
                data,
                nominal_passive=learner,
                include_constraint=True,
                dof_index=0,
                qvel_for_passive=qvel_pre,
            )[0]
        )
        qvel[i] = float(qvel_pre[0])
        applied[i] = float(tau)
        r_oracle[i] = -delta_b * float(qvel_pre[0])

    return {
        "qvel": qvel,
        "residual": residual,
        "qfrc_applied": applied,
        "r_oracle": r_oracle,
        "delta_b": np.array([delta_b], dtype=np.float64),
    }


def _steady_mask(n_steps: int, dt: float, t0: float) -> np.ndarray:
    t = np.arange(n_steps, dtype=np.float64) * dt
    return t >= float(t0)


def evaluate_action_channel(
    config: SIMX2Config,
    *,
    action: str,
    seed: int,
) -> dict[str, Any]:
    n_steps = int(round(config.duration_s / config.timestep))
    torques = action_torque_sequence(
        action, n_steps=n_steps, dt=config.timestep
    )
    by_b: dict[float, dict[str, np.ndarray]] = {}
    for b in config.true_dampings:
        by_b[float(b)] = _rollout_action_damping(
            torques=torques,
            timestep=config.timestep,
            true_damping=float(b),
            seed=seed,
        )

    # G1: identical applied sequence across plants.
    applied_ref = by_b[0.10]["qfrc_applied"]
    applied_ok = all(
        bool(np.allclose(by_b[float(b)]["qfrc_applied"], applied_ref, atol=config.applied_match_atol))
        for b in config.true_dampings
    )
    applied_ok = applied_ok and bool(
        np.allclose(applied_ref, torques, atol=config.applied_match_atol)
    )

    # G0: oracle residual reconstruction on mismatch plants.
    e_oracles: list[float] = []
    for b in (0.05, 0.15):
        r = by_b[b]["residual"]
        r_o = by_b[b]["r_oracle"]
        rmse = float(np.sqrt(np.mean(np.square(r - r_o))))
        rms = float(np.sqrt(np.mean(np.square(r_o))))
        e_oracles.append(rmse / (rms + EPS))
    e_oracle = float(max(e_oracles))

    mask = _steady_mask(n_steps, config.timestep, config.steady_t0_s)
    qvel_info = {b: by_b[float(b)]["qvel"][mask] for b in config.true_dampings}
    # Use nominal plant velocity for E_v (same torque; small damping difference).
    e_v = float(np.mean(np.square(qvel_info[0.10])))
    r05 = by_b[0.05]["residual"][mask]
    r15 = by_b[0.15]["residual"][mask]
    r10 = by_b[0.10]["residual"][mask]
    abs_r_mismatch = 0.5 * (np.abs(r05) + np.abs(r15))
    mean_abs_r = float(np.mean(abs_r_mismatch))

    mi_t = np.empty(int(np.count_nonzero(mask)), dtype=np.float64)
    for i, (m05, m15) in enumerate(zip(r05, r15, strict=True)):
        # Generative valid mean is 0 by prereg (assumption validity), not r10.
        mi_t[i] = mi_binary_v_given_mixture(
            mean_valid=0.0,
            mean_inv_a=float(m05),
            mean_inv_b=float(m15),
            sigma=config.sigma_obs,
        )
    i_a = float(np.mean(mi_t))

    # Nominal NRMSE for diagnostics (not a primary X2 gate).
    nrmse_nom = residual_nrmse(
        by_b[0.10]["residual"][:, None], by_b[0.10]["qfrc_applied"][:, None]
    )

    return {
        "action": action,
        "seed": seed,
        "i_a_bits": i_a,
        "e_v": e_v,
        "mean_abs_r_mismatch": mean_abs_r,
        "e_oracle": e_oracle,
        "g0_oracle": bool(e_oracle < config.oracle_tol),
        "g1_applied": applied_ok,
        "nrmse_nominal": float(nrmse_nom),
        "torque_rms": float(np.sqrt(np.mean(np.square(torques)))),
        "steady_n": int(np.count_nonzero(mask)),
        "mi_t_mean": i_a,
        "mi_t_std": float(np.std(mi_t)),
        "arrays": {
            "qfrc_applied": applied_ref,
            "qvel_b0.10": by_b[0.10]["qvel"],
            "residual_b0.05": by_b[0.05]["residual"],
            "residual_b0.10": by_b[0.10]["residual"],
            "residual_b0.15": by_b[0.15]["residual"],
            "r_oracle_b0.05": by_b[0.05]["r_oracle"],
            "r_oracle_b0.15": by_b[0.15]["r_oracle"],
            "mi_t_steady": mi_t,
        },
    }


def run_sim_x2(
    output: str | Path,
    *,
    config: SIMX2Config | None = None,
    x1_summary: str | Path | None = "runs/sim_x1/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or SIMX2Config()
    if abs(cfg.learner_damping - SIMX_DAMPING) > 1.0e-12:
        raise RuntimeError("SIM-X2 learner_damping must remain frozen at 0.10")
    if abs(cfg.sigma_obs - 0.01) > 1.0e-12:
        raise RuntimeError("SIM-X2 forbids retuning sigma_obs away from 0.01")
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("SIM-X2 must not write under runs/r10_c0/")
    if x1_summary is not None:
        path = Path(x1_summary)
        if not path.is_file():
            raise RuntimeError(f"SIM-X2 locked until SIM-X1 summary exists ({path})")
        x1 = json.loads(path.read_text(encoding="utf-8"))
        if not bool(x1.get("sim_x1_passed")):
            raise RuntimeError("SIM-X2 locked until sim_x1_passed=true")
        if x1.get("host_plant_id") != HOST_PLANT_ID:
            raise RuntimeError("SIM-X2 requires X1 host_plant_id=simx_hinge.v1")

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for action in SIMX2_ACTIONS:
        for seed in cfg.seeds:
            cell = evaluate_action_channel(cfg, action=action, seed=seed)
            rel = f"action_{action}/seed_{seed}.h5"
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            arrays = cell.pop("arrays")
            with h5py.File(out, "w") as handle:
                handle.attrs["schema"] = SCHEMA_ID
                handle.attrs["source"] = "simulator"
                handle.attrs["label"] = "sim-x2"
                handle.attrs["host_plant_id"] = HOST_PLANT_ID
                handle.attrs["real_physics"] = False
                handle.attrs["sigma_obs"] = cfg.sigma_obs
                handle.attrs["sigma_obs_kind"] = "synthetic_observation_resolution_surrogate"
                handle.attrs["action"] = action
                handle.attrs["seed"] = seed
                handle.attrs["i_a_bits"] = cell["i_a_bits"]
                handle.attrs["e_v"] = cell["e_v"]
                meta = handle.create_group("metadata")
                meta.attrs["json"] = json.dumps(cell, sort_keys=True, default=str)
                for key, value in arrays.items():
                    handle.create_dataset(key, data=value, compression="gzip")
            cell["path"] = str(out)
            rows.append(cell)

    def _mean(action: str, key: str) -> float:
        vals = [r[key] for r in rows if r["action"] == action]
        return float(np.mean(vals))

    i_cons = _mean("cons", "i_a_bits")
    i_info = _mean("info", "i_a_bits")
    i_low = _mean("low_f", "i_a_bits")
    i_high = _mean("high_f", "i_a_bits")
    e_cons = _mean("cons", "e_v")
    e_info = _mean("info", "e_v")
    e_low = _mean("low_f", "e_v")
    e_high = _mean("high_f", "e_v")

    g0 = all(bool(r["g0_oracle"]) for r in rows)
    g1 = all(bool(r["g1_applied"]) for r in rows)
    g2 = bool(i_info > i_cons and (i_info - i_cons) > cfg.i_gap_min)
    g3 = bool(i_cons < cfg.i_cons_max and i_info > cfg.i_info_min)
    g4 = bool(e_low > e_high and (i_low - i_high) > cfg.i_gap_min)
    passed = bool(g0 and g1 and g2 and g3 and g4)

    # info ≡ low_f waveform: I and E_v must match up to seed average noise of MI grid.
    info_low_match = bool(
        abs(i_info - i_low) < 1.0e-9 and abs(e_info - e_low) < 1.0e-12
    )

    mujoco = import_mujoco()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "SIM-X2",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": "action-conditioned validity information channel on MuJoCo",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "unlocks_sim_x3": passed,
        "sigma_obs": cfg.sigma_obs,
        "sigma_obs_kind": "synthetic_observation_resolution_surrogate",
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "actions": SIMX2_ACTIONS,
        "I_bits_per_sample": {
            "cons": i_cons,
            "info": i_info,
            "low_f": i_low,
            "high_f": i_high,
            "info_minus_cons": i_info - i_cons,
            "low_f_minus_high_f": i_low - i_high,
        },
        "E_v": {
            "cons": e_cons,
            "info": e_info,
            "low_f": e_low,
            "high_f": e_high,
        },
        "mean_abs_r_mismatch": {
            name: _mean(name, "mean_abs_r_mismatch") for name in SIMX2_ACTIONS
        },
        "info_equals_low_f": info_low_match,
        "cells": [{k: v for k, v in r.items() if k != "arrays"} for r in rows],
        "cell_count": len(rows),
        "g0_x1_carryover": g0,
        "g1_action_isolation": g1,
        "g2_primary_ordering": g2,
        "g3_blindness": g3,
        "g4_iso_energy": g4,
        "sim_x2_passed": passed,
        "claims_if_pass": {
            "action_conditioned_observability": bool(g2 and g4),
            "policy_induced_blindness_instance": bool(g3),
            "cross_engine_only": True,
        },
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
