"""SIM-X0: nominal force accounting on the SIM-X host plant.

Does not unlock R10-C0. Does not run validity, certificates, or X1.
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
    known_control_force,
    nominal_passive_force,
    residual_nrmse,
)
from .mujoco_physics import import_mujoco
from .simx_plant import (
    HOST_PLANT_ID,
    assert_x0_plant,
    load_simx_hinge,
    nominal_passive_params,
    simx_hinge_xml,
    simx_hinge_xml_sha256,
)

PREREG_PATH = "REPORT/REG/SIMX/SIMX0_PREREG.md"
ACCOUNTING_PATH = "REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md"
SCHEMA_ID = "aprwm.sim_x0.residual.v1"
NRMSE_MAX = 1.0e-4
CONSTRAINT_ABS_MAX = 1.0e-6

SIMX0_SEEDS = (9101, 9111, 9121)
SIMX0_TRAJECTORIES = ("sine", "chirp", "piecewise")


@dataclass(frozen=True)
class SIMX0Config:
    seeds: tuple[int, ...] = SIMX0_SEEDS
    trajectories: tuple[str, ...] = SIMX0_TRAJECTORIES
    duration_s: float = 10.0
    timestep: float = 0.002
    nrmse_max: float = NRMSE_MAX
    constraint_abs_max: float = CONSTRAINT_ABS_MAX
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


def _torque_sequence(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    config: SIMX0Config,
) -> np.ndarray:
    t = np.arange(n_steps, dtype=np.float64) * dt
    if kind == "sine":
        return config.sine_amplitude * np.sin(2.0 * np.pi * config.sine_freq_hz * t)
    if kind == "chirp":
        f0 = config.chirp_f0_hz
        f1 = config.chirp_f1_hz
        k = (f1 - f0) / max(config.duration_s, dt)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return config.chirp_amplitude * np.sin(phase)
    if kind == "piecewise":
        rng = np.random.default_rng(seed + 17_000)
        hold = max(1, int(round(config.piecewise_hold_s / dt)))
        values: list[float] = []
        while len(values) < n_steps:
            level = float(
                rng.uniform(-config.piecewise_amplitude, config.piecewise_amplitude)
            )
            values.extend([level] * hold)
        return np.asarray(values[:n_steps], dtype=np.float64)
    raise ValueError(f"Unknown trajectory kind: {kind}")


def _locus(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    residual = np.asarray(arrays["residual"], dtype=np.float64).reshape(-1)
    constraint = np.asarray(arrays["qfrc_constraint"], dtype=np.float64).reshape(-1)
    m_qacc = np.asarray(arrays["M_qacc"], dtype=np.float64).reshape(-1)
    known = np.asarray(arrays["qfrc_applied"], dtype=np.float64).reshape(-1)
    passive = np.asarray(arrays["tau_nominal_passive"], dtype=np.float64).reshape(-1)
    bias = np.asarray(arrays["qfrc_bias"], dtype=np.float64).reshape(-1)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or float(np.std(a)) < 1.0e-18 or float(np.std(b)) < 1.0e-18:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    mean_r = float(np.mean(residual))
    std_r = float(np.std(residual))
    max_abs_r = float(np.max(np.abs(residual)))
    max_abs_c = float(np.max(np.abs(constraint)))
    hints: list[str] = []
    if max_abs_c > CONSTRAINT_ABS_MAX:
        hints.append("constraint")
    if abs(mean_r) > 10.0 * max(std_r, 1.0e-16):
        hints.append("bias")
    if abs(_corr(residual, m_qacc)) > 0.8 and max_abs_r > 1.0e-6:
        hints.append("sign")
    if abs(_corr(residual, passive)) > 0.8 and max_abs_r > 1.0e-6:
        hints.append("passive")
    order = ("sign", "bias", "passive", "constraint")
    ranked = [name for name in order if name in hints]
    return {
        "nrmse_tau": None,
        "max_abs_residual": max_abs_r,
        "mean_residual": mean_r,
        "std_residual": std_r,
        "max_abs_qfrc_constraint": max_abs_c,
        "corr_residual_M_qacc": _corr(residual, m_qacc),
        "corr_residual_qfrc_applied": _corr(residual, known),
        "corr_residual_nominal_passive": _corr(residual, passive),
        "corr_residual_qfrc_bias": _corr(residual, bias),
        "atlas_order": list(order),
        "atlas_hits": ranked,
        "note": "If G-close fails, walk atlas_order; do not add a network.",
    }


def _rollout_cell(config: SIMX0Config, *, seed: int, trajectory: str) -> dict[str, Any]:
    mujoco = import_mujoco()
    model, data = load_simx_hinge(config.timestep)
    inventory = assert_x0_plant(model)
    n_steps = int(round(config.duration_s / config.timestep))
    torques = _torque_sequence(
        trajectory, n_steps=n_steps, dt=config.timestep, seed=seed, config=config
    )
    nominal = nominal_passive_params()

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
        "qfrc_passive": [],
    }

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.15 * np.sin(seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    for tau in torques:
        # Passive / bias / M are evaluated at the *pre-step* state. After
        # mj_step, qpos/qvel advance but qfrc_* and qacc still refer to that
        # evaluation. Using post-step qvel in NominalPassiveParams is a
        # timing interface bug (invisible when damping=0, as in R1-MJ0).
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
            nominal_passive=nominal,
            include_constraint=True,
            dof_index=0,
            qvel_for_passive=qvel_pre,
        )
        records["qpos"].append(qpos_pre)
        records["qvel"].append(qvel_pre)
        records["qacc"].append(qacc)
        records["M_qacc"].append(mass @ qacc)
        records["qfrc_bias"].append(get_bias_force(data))
        records["qfrc_applied"].append(get_applied_force(data))
        records["qfrc_actuator"].append(get_actuator_force(data))
        records["qfrc_constraint"].append(get_constraint_force(data))
        records["tau_nominal_passive"].append(
            nominal_passive_force(qvel_pre, nominal, dof_index=0)
        )
        records["residual"].append(residual)
        records["qfrc_passive"].append(get_truth_passive_force(data))

    packed = {key: np.stack(vals, axis=0) for key, vals in records.items()}
    nrmse = residual_nrmse(packed["residual"], packed["qfrc_applied"])
    locus = _locus(packed)
    locus["nrmse_tau"] = float(nrmse)
    close = bool(nrmse < config.nrmse_max)
    constraint_ok = bool(locus["max_abs_qfrc_constraint"] < config.constraint_abs_max)
    actuator_ok = bool(np.max(np.abs(packed["qfrc_actuator"])) < 1.0e-12)
    return {
        "seed": seed,
        "trajectory": trajectory,
        "nrmse": float(nrmse),
        "close": close,
        "constraint_ok": constraint_ok,
        "actuator_ok": actuator_ok,
        "g_close": close,
        "host_plant_id": HOST_PLANT_ID,
        "xml_sha256": simx_hinge_xml_sha256(config.timestep),
        "plant_inventory": inventory,
        "n_steps": n_steps,
        "timestep": config.timestep,
        "locus": locus,
        "arrays": packed,
    }


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = SCHEMA_ID
        handle.attrs["source"] = "simulator"
        handle.attrs["label"] = "sim-x0"
        handle.attrs["host_plant_id"] = HOST_PLANT_ID
        handle.attrs["real_physics"] = False
        handle.attrs["unlocks_r10_c0"] = False
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
        ):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_truth_qfrc_passive"] = True
        raw = handle.create_group("raw_truth")
        raw.create_dataset("qfrc_passive", data=arrays["qfrc_passive"], compression="gzip")
        raw.attrs["audit_only"] = True
        evaluation = handle.create_group("evaluation")
        evaluation.attrs["nrmse_tau"] = cell["nrmse"]
        evaluation.attrs["max_abs_residual"] = cell["locus"]["max_abs_residual"]
        evaluation.attrs["mean_residual"] = cell["locus"]["mean_residual"]
        evaluation.attrs["std_residual"] = cell["locus"]["std_residual"]
        evaluation.attrs["max_abs_qfrc_constraint"] = cell["locus"][
            "max_abs_qfrc_constraint"
        ]
        evaluation.attrs["close"] = cell["close"]


def all_cells_individually_close(rows: list[dict[str, Any]]) -> bool:
    required = {(seed, traj) for seed in SIMX0_SEEDS for traj in SIMX0_TRAJECTORIES}
    observed = {(row["seed"], row["trajectory"]) for row in rows}
    return observed == required and all(bool(row["close"]) for row in rows)


def run_sim_x0(
    output: str | Path,
    *,
    config: SIMX0Config | None = None,
) -> dict[str, Any]:
    cfg = config or SIMX0Config()
    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("SIM-X0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)
    (root / "simx_hinge.xml").write_text(simx_hinge_xml(cfg.timestep), encoding="utf-8")

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
                    "nrmse": cell["nrmse"],
                    "close": cell["close"],
                    "constraint_ok": cell["constraint_ok"],
                    "actuator_ok": cell["actuator_ok"],
                    "max_abs_residual": cell["locus"]["max_abs_residual"],
                    "mean_residual": cell["locus"]["mean_residual"],
                    "std_residual": cell["locus"]["std_residual"],
                    "max_abs_qfrc_constraint": cell["locus"]["max_abs_qfrc_constraint"],
                    "atlas_hits": cell["locus"]["atlas_hits"],
                }
            )

    g_close = all_cells_individually_close(rows)
    g_constraint = all(bool(row["constraint_ok"]) for row in rows)
    g_actuator = all(bool(row["actuator_ok"]) for row in rows)
    passed = bool(g_close and g_constraint and g_actuator)
    mujoco = import_mujoco()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    nrmses = [float(row["nrmse"]) for row in rows]
    summary = {
        "stage": "SIM-X0",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "xml_sha256": simx_hinge_xml_sha256(cfg.timestep),
        "prereg": PREREG_PATH,
        "accounting": ACCOUNTING_PATH,
        "scientific_claim": "cross-engine mechanism robustness preflight only",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "unlocks_sim_x1": passed,
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "nrmse_max": cfg.nrmse_max,
        "cells": rows,
        "cell_count": len(rows),
        "max_nrmse_tau": max(nrmses) if nrmses else None,
        "g_close": g_close,
        "g_constraint": g_constraint,
        "g_actuator": g_actuator,
        "g_engine": True,
        "g_label": True,
        "sim_x0_passed": passed,
        "pass_iff": "same-host-plant nominal force accounting closes below 1e-4",
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
