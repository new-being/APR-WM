"""SIM-X1: oracle damping mismatch on the SIM-X host plant.

Learner nominal damping stays frozen at b0=0.10. Plant dof_damping may
differ. Does not unlock R10-C0 or SIM-X2 language.
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
    get_actuator_force,
    get_applied_force,
    get_bias_force,
    get_constraint_force,
    get_mass_matrix,
    get_truth_passive_force,
    nominal_passive_force,
    residual_nrmse,
)
from .mujoco_physics import import_mujoco
from .sim_x0 import SIMX0Config, _torque_sequence
from .simx_plant import (
    HOST_PLANT_ID,
    SIMX_DAMPING,
    assert_x1_plant,
    learner_nominal_params,
    load_simx_hinge,
    simx_hinge_xml,
    simx_hinge_xml_sha256,
)

PREREG_PATH = "REPORT/REG/SIMX/SIMX1_PREREG.md"
SCHEMA_ID = "aprwm.sim_x1.residual.v1"
EPS = 1.0e-8
GATE_TOL = 1.0e-4

SIMX1_SEEDS = (9101, 9111, 9121)
SIMX1_TRAJECTORIES = ("sine", "chirp", "piecewise")
SIMX1_TRUE_DAMPINGS = (0.05, 0.10, 0.15)


@dataclass(frozen=True)
class SIMX1Config:
    seeds: tuple[int, ...] = SIMX1_SEEDS
    trajectories: tuple[str, ...] = SIMX1_TRAJECTORIES
    true_dampings: tuple[float, ...] = SIMX1_TRUE_DAMPINGS
    learner_damping: float = SIMX_DAMPING
    duration_s: float = 10.0
    timestep: float = 0.002
    gate_tol: float = GATE_TOL
    sine_amplitude: float = 1.5
    sine_freq_hz: float = 0.4
    chirp_amplitude: float = 1.2
    chirp_f0_hz: float = 0.1
    chirp_f1_hz: float = 2.0
    piecewise_amplitude: float = 1.8
    piecewise_hold_s: float = 0.5


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _as_x0_excitation(config: SIMX1Config) -> SIMX0Config:
    return SIMX0Config(
        seeds=config.seeds,
        trajectories=config.trajectories,
        duration_s=config.duration_s,
        timestep=config.timestep,
        sine_amplitude=config.sine_amplitude,
        sine_freq_hz=config.sine_freq_hz,
        chirp_amplitude=config.chirp_amplitude,
        chirp_f0_hz=config.chirp_f0_hz,
        chirp_f1_hz=config.chirp_f1_hz,
        piecewise_amplitude=config.piecewise_amplitude,
        piecewise_hold_s=config.piecewise_hold_s,
    )


def _fit_slope(qvel: np.ndarray, residual: np.ndarray) -> float:
    v = np.asarray(qvel, dtype=np.float64).reshape(-1)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    denom = float(np.dot(v, v))
    if denom < EPS:
        return 0.0
    return float(np.dot(v, r) / denom)


def _fit_slope_intercept(qvel: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    v = np.asarray(qvel, dtype=np.float64).reshape(-1)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    n = float(v.size)
    sum_v = float(np.sum(v))
    sum_r = float(np.sum(r))
    sum_vv = float(np.dot(v, v))
    sum_vr = float(np.dot(v, r))
    det = n * sum_vv - sum_v * sum_v
    if abs(det) < EPS:
        return _fit_slope(v, r), float(np.mean(r))
    slope = (n * sum_vr - sum_v * sum_r) / det
    intercept = (sum_r - slope * sum_v) / n
    return float(slope), float(intercept)


def _rollout_cell(
    config: SIMX1Config,
    *,
    seed: int,
    trajectory: str,
    true_damping: float,
) -> dict[str, Any]:
    mujoco = import_mujoco()
    b_true = float(true_damping)
    b0 = float(config.learner_damping)
    if abs(b0 - SIMX_DAMPING) > 1.0e-12:
        raise RuntimeError("SIM-X1 forbids retuning learner_damping away from b0=0.10")

    model, data = load_simx_hinge(config.timestep, true_damping=b_true)
    inventory = assert_x1_plant(model, true_damping=b_true)
    learner = learner_nominal_params()
    if abs(learner.damping - b0) > 1.0e-12:
        raise RuntimeError("learner_nominal_params leaked away from frozen b0")

    n_steps = int(round(config.duration_s / config.timestep))
    torques = _torque_sequence(
        trajectory,
        n_steps=n_steps,
        dt=config.timestep,
        seed=seed,
        config=_as_x0_excitation(config),
    )
    delta_b = b_true - b0

    records: dict[str, list[np.ndarray]] = {
        "qpos": [],
        "qvel": [],
        "qacc": [],
        "M_qacc": [],
        "qfrc_bias": [],
        "qfrc_applied": [],
        "qfrc_actuator": [],
        "qfrc_constraint": [],
        "tau_nominal_passive": [],
        "residual": [],
        "r_oracle": [],
        "qfrc_passive": [],
    }

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.15 * np.sin(seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    for tau in torques:
        qpos_pre = np.array(data.qpos, dtype=np.float64, copy=True)
        qvel_pre = np.array(data.qvel, dtype=np.float64, copy=True)
        data.qfrc_applied[:] = 0.0
        data.qfrc_actuator[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        mass = get_mass_matrix(model, data)
        qacc = np.array(data.qacc, dtype=np.float64, copy=True)
        residual = generalized_force_residual(
            model,
            data,
            nominal_passive=learner,
            include_constraint=True,
            dof_index=0,
            qvel_for_passive=qvel_pre,
        )
        r_oracle = -delta_b * qvel_pre
        records["qpos"].append(qpos_pre)
        records["qvel"].append(qvel_pre)
        records["qacc"].append(qacc)
        records["M_qacc"].append(mass @ qacc)
        records["qfrc_bias"].append(get_bias_force(data))
        records["qfrc_applied"].append(get_applied_force(data))
        records["qfrc_actuator"].append(get_actuator_force(data))
        records["qfrc_constraint"].append(get_constraint_force(data))
        records["tau_nominal_passive"].append(
            nominal_passive_force(qvel_pre, learner, dof_index=0)
        )
        records["residual"].append(residual)
        records["r_oracle"].append(r_oracle)
        records["qfrc_passive"].append(get_truth_passive_force(data))

    packed = {key: np.stack(vals, axis=0) for key, vals in records.items()}
    residual = packed["residual"].reshape(-1)
    r_oracle = packed["r_oracle"].reshape(-1)
    qvel = packed["qvel"].reshape(-1)
    tau_known = packed["qfrc_applied"].reshape(-1)

    nrmse = residual_nrmse(packed["residual"], packed["qfrc_applied"])
    rmse_err = float(np.sqrt(np.mean(np.square(residual - r_oracle))))
    rms_oracle = float(np.sqrt(np.mean(np.square(r_oracle))))
    e_oracle = rmse_err / (rms_oracle + EPS)
    s_hat = _fit_slope(qvel, residual)
    s_star = -delta_b
    if abs(s_star) < EPS:
        s_rel = abs(s_hat - s_star)  # nominal cell: slope ~ 0
    else:
        s_rel = abs(s_hat - s_star) / abs(s_star)
    s2, c_hat = _fit_slope_intercept(qvel, residual)
    c_rel = abs(c_hat) / (float(np.sqrt(np.mean(np.square(tau_known)))) + EPS)

    is_nominal = abs(delta_b) < 1.0e-12
    is_mismatch = not is_nominal
    g0 = (not is_nominal) or bool(nrmse < config.gate_tol)
    g1 = (not is_mismatch) or bool(e_oracle < config.gate_tol)
    g2 = (not is_mismatch) or bool(s_rel < config.gate_tol)
    g3 = bool(c_rel < config.gate_tol)

    return {
        "seed": seed,
        "trajectory": trajectory,
        "true_damping": b_true,
        "learner_damping": b0,
        "delta_b": delta_b,
        "true_plant_params": {"damping": b_true},
        "learner_nominal_params": {"damping": b0},
        "nrmse_tau": float(nrmse),
        "e_oracle": float(e_oracle),
        "s_hat": float(s_hat),
        "s_star": float(s_star),
        "s_rel_err": float(s_rel),
        "s_hat_with_intercept": float(s2),
        "c_hat": float(c_hat),
        "c_rel": float(c_rel),
        "g0": g0,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "cell_pass": bool(g0 and g1 and g2 and g3),
        "host_plant_id": HOST_PLANT_ID,
        "xml_sha256": simx_hinge_xml_sha256(config.timestep, true_damping=b_true),
        "plant_inventory": inventory,
        "n_steps": n_steps,
        "timestep": config.timestep,
        "arrays": packed,
    }


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "sim-x1"
        handle.attrs["host_plant_id"] = HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["unlocks_r10_c0"] = False
        handle.attrs["true_damping"] = cell["true_damping"]
        handle.attrs["learner_damping"] = cell["learner_damping"]
        meta_grp = handle.create_group("metadata")
        meta_grp.attrs["json"] = json.dumps(meta, sort_keys=True, default=str)
        learner = handle.create_group("learner_visible")
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "M_qacc",
            "qfrc_bias",
            "qfrc_applied",
            "qfrc_actuator",
            "qfrc_constraint",
            "tau_nominal_passive",
            "residual",
            "r_oracle",
        ):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_truth_qfrc_passive"] = True
        learner.attrs["learner_damping"] = cell["learner_damping"]
        raw = handle.create_group("raw_truth")
        raw.create_dataset("qfrc_passive", data=arrays["qfrc_passive"], compression="gzip")
        raw.attrs["audit_only"] = True
        raw.attrs["true_damping"] = cell["true_damping"]
        evaluation = handle.create_group("evaluation")
        for key in (
            "nrmse_tau",
            "e_oracle",
            "s_hat",
            "s_star",
            "s_rel_err",
            "c_hat",
            "c_rel",
            "g0",
            "g1",
            "g2",
            "g3",
            "cell_pass",
        ):
            evaluation.attrs[key] = cell[key]


def all_cells_present(rows: list[dict[str, Any]], config: SIMX1Config) -> bool:
    required = {
        (seed, traj, float(b))
        for seed in config.seeds
        for traj in config.trajectories
        for b in config.true_dampings
    }
    observed = {
        (row["seed"], row["trajectory"], float(row["true_damping"])) for row in rows
    }
    return observed == required


def run_sim_x1(
    output: str | Path,
    *,
    config: SIMX1Config | None = None,
    x0_summary: str | Path | None = "runs/sim_x0/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or SIMX1Config()
    if abs(cfg.learner_damping - SIMX_DAMPING) > 1.0e-12:
        raise RuntimeError("SIM-X1 learner_damping must remain frozen at 0.10")
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("SIM-X1 must not write under runs/r10_c0/")
    if x0_summary is not None:
        x0_path = Path(x0_summary)
        if not x0_path.is_file():
            raise RuntimeError(f"SIM-X1 locked until SIM-X0 summary exists ({x0_path})")
        x0 = json.loads(x0_path.read_text(encoding="utf-8"))
        if not bool(x0.get("sim_x0_passed")):
            raise RuntimeError("SIM-X1 locked until sim_x0_passed=true")
        if x0.get("host_plant_id") != HOST_PLANT_ID:
            raise RuntimeError("SIM-X1 requires X0 host_plant_id=simx_hinge.v1")

    root.mkdir(parents=True, exist_ok=True)
    for b_true in cfg.true_dampings:
        rel = f"plant_b{b_true:.2f}/simx_hinge.xml"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            simx_hinge_xml(cfg.timestep, true_damping=b_true), encoding="utf-8"
        )

    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for trajectory in cfg.trajectories:
            for b_true in cfg.true_dampings:
                cell = _rollout_cell(
                    cfg, seed=seed, trajectory=trajectory, true_damping=b_true
                )
                rel = f"plant_b{b_true:.2f}/seed_{seed}/{trajectory}.h5"
                _write_cell_h5(root / rel, cell)
                rows.append(
                    {
                        "seed": seed,
                        "trajectory": trajectory,
                        "true_damping": b_true,
                        "learner_damping": cfg.learner_damping,
                        "delta_b": cell["delta_b"],
                        "path": str(root / rel),
                        "nrmse_tau": cell["nrmse_tau"],
                        "e_oracle": cell["e_oracle"],
                        "s_hat": cell["s_hat"],
                        "s_star": cell["s_star"],
                        "s_rel_err": cell["s_rel_err"],
                        "c_hat": cell["c_hat"],
                        "c_rel": cell["c_rel"],
                        "g0": cell["g0"],
                        "g1": cell["g1"],
                        "g2": cell["g2"],
                        "g3": cell["g3"],
                        "cell_pass": cell["cell_pass"],
                    }
                )

    if not all_cells_present(rows, cfg):
        raise RuntimeError("SIM-X1 matrix incomplete")

    nominal_rows = [r for r in rows if abs(r["delta_b"]) < 1.0e-12]
    mismatch_rows = [r for r in rows if abs(r["delta_b"]) >= 1.0e-12]
    g0 = all(bool(r["g0"]) for r in nominal_rows)
    g1 = all(bool(r["g1"]) for r in mismatch_rows)
    g2 = all(bool(r["g2"]) for r in mismatch_rows)
    g3 = all(bool(r["g3"]) for r in rows)
    passed = bool(g0 and g1 and g2 and g3 and all(r["cell_pass"] for r in rows))

    mujoco = import_mujoco()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "SIM-X1",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": "oracle damping mismatch → correct force-space residual",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "unlocks_sim_x2": passed,
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "true_plant_params_grid": {"damping": list(cfg.true_dampings)},
        "learner_nominal_params": {"damping": cfg.learner_damping},
        "anti_leakage": "learner_nominal never reads plant dof_damping",
        "gate_tol": cfg.gate_tol,
        "cells": rows,
        "cell_count": len(rows),
        "max_nrmse_nominal": max((r["nrmse_tau"] for r in nominal_rows), default=None),
        "max_e_oracle_mismatch": max((r["e_oracle"] for r in mismatch_rows), default=None),
        "max_s_rel_err_mismatch": max(
            (r["s_rel_err"] for r in mismatch_rows), default=None
        ),
        "max_c_rel": max((r["c_rel"] for r in rows), default=None),
        "g0_nominal_guard": g0,
        "g1_oracle_reconstruction": g1,
        "g2_signed_coefficient": g2,
        "g3_no_spurious_intercept": g3,
        "sim_x1_passed": passed,
        "pass_iff": "oracle damping mismatch closes under frozen b0=0.10",
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
