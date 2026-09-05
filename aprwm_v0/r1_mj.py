"""R1-MJ0: native MuJoCo single-hinge force-space closure.

Infrastructure gate only.  Does not unlock scientific revision claims.
R1-MS0 remains frozen as ``infrastructure_block`` and is not evaluated here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from .mujoco_force import (
    NominalPassiveParams,
    generalized_force_residual,
    get_applied_force,
    get_bias_force,
    get_constraint_force,
    get_mass_matrix,
    get_truth_passive_force,
    known_control_force,
    residual_nrmse,
)
from .mujoco_physics import import_mujoco


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


MJ0_SEEDS = (8201, 8211, 8221)
MJ0_TRAJECTORIES = ("sine", "chirp", "piecewise")
MJ0_DURATION_S = 10.0
MJ0_TIMESTEP = 0.002
MJ0_NRMSE_MAX = 1.0e-4
HORIZONS = (2, 4, 8, 32)
ACCEPTANCE_HORIZONS = (2, 4, 8)

HINGE_MJCF = """
<mujoco model="r1_mj0_hinge">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <joint limited="true" damping="0" frictionloss="0" armature="0"/>
    <geom rgba="0.7 0.7 0.8 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.2 0.2 0.2 1"/>
    <body name="door" pos="0 0 0.5">
      <inertial pos="0.15 0 0" mass="2.0" diaginertia="0.05 0.08 0.05"/>
      <joint name="hinge" type="hinge" axis="0 0 1"
             range="-1.5708 1.5708" damping="0" frictionloss="0"/>
      <geom name="door_panel" type="box" size="0.2 0.02 0.3" pos="0.2 0 0"
            density="200"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


@dataclass(frozen=True)
class R1MJ0Config:
    seeds: tuple[int, ...] = MJ0_SEEDS
    trajectories: tuple[str, ...] = MJ0_TRAJECTORIES
    duration_s: float = MJ0_DURATION_S
    timestep: float = MJ0_TIMESTEP
    nrmse_max: float = MJ0_NRMSE_MAX
    sine_amplitude: float = 1.5
    sine_freq_hz: float = 0.4
    chirp_amplitude: float = 1.2
    chirp_f0_hz: float = 0.1
    chirp_f1_hz: float = 2.0
    piecewise_amplitude: float = 1.8
    piecewise_hold_s: float = 0.5


def stage_status() -> dict[str, Any]:
    return {
        "R1-MS0": {
            "frozen": True,
            "status": "infrastructure_block",
            "scientific_no_go": False,
            "note": "ManiSkill Drawer retained in record; not a failed experiment",
        },
        "R1-MJ/RS": {
            "selected_as_next_realism_bridge": True,
            "definition": (
                "Frozen R0.6 under MuJoCo dynamics, robot contact, and task diversity"
            ),
        },
        "R1-MJ0": {"unlocked": True},
        "R1-RS0": {"locked_until_MJ0_passes": True},
        "R1-RS1": {"locked": True},
        "R1-RS2": {"locked": True},
        "Wipe_ToolHang": {"locked": True},
    }


def _xml_sha256(timestep: float) -> str:
    xml = HINGE_MJCF.format(timestep=timestep).encode("utf-8")
    return hashlib.sha256(xml).hexdigest()


def _make_model_data(config: R1MJ0Config):
    mujoco = import_mujoco()

    xml = HINGE_MJCF.format(timestep=config.timestep)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def _torque_sequence(
    kind: str,
    *,
    n_steps: int,
    dt: float,
    seed: int,
    config: R1MJ0Config,
) -> np.ndarray:
    t = np.arange(n_steps, dtype=np.float64) * dt
    if kind == "sine":
        return config.sine_amplitude * np.sin(2.0 * np.pi * config.sine_freq_hz * t)
    if kind == "chirp":
        # Linear chirp phase integral.
        f0 = config.chirp_f0_hz
        f1 = config.chirp_f1_hz
        k = (f1 - f0) / max(config.duration_s, dt)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return config.chirp_amplitude * np.sin(phase)
    if kind == "piecewise":
        rng = np.random.default_rng(seed + 17_000)
        hold = max(1, int(round(config.piecewise_hold_s / dt)))
        values = []
        while len(values) < n_steps:
            level = float(rng.uniform(-config.piecewise_amplitude, config.piecewise_amplitude))
            values.extend([level] * hold)
        return np.asarray(values[:n_steps], dtype=np.float64)
    raise ValueError(f"Unknown trajectory kind: {kind}")


def _snapshot(model, data) -> dict[str, np.ndarray]:
    return {
        "qpos": np.array(data.qpos, copy=True),
        "qvel": np.array(data.qvel, copy=True),
        "ctrl": np.array(data.ctrl, copy=True),
        "time": np.array([data.time], dtype=np.float64),
        "qfrc_applied": np.array(data.qfrc_applied, copy=True),
    }


def _restore(model, data, snap: dict[str, np.ndarray]) -> None:
    mujoco = import_mujoco()

    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.ctrl[:] = snap["ctrl"]
    data.qfrc_applied[:] = snap["qfrc_applied"]
    data.time = float(snap["time"][0])
    mujoco.mj_forward(model, data)


def _rollout_cell(
    config: R1MJ0Config,
    *,
    seed: int,
    trajectory: str,
) -> dict[str, Any]:
    mujoco = import_mujoco()

    model, data = _make_model_data(config)
    n_steps = int(round(config.duration_s / config.timestep))
    torques = _torque_sequence(
        trajectory, n_steps=n_steps, dt=config.timestep, seed=seed, config=config
    )
    nominal = NominalPassiveParams(damping=0.0, frictionloss=0.0, armature=0.0)

    records: dict[str, list[np.ndarray]] = {
        "qpos": [],
        "qvel": [],
        "qacc": [],
        "tau_applied": [],
        "qfrc_bias": [],
        "qfrc_constraint": [],
        "qfrc_passive_truth": [],
        "residual": [],
        "known_control": [],
    }

    mujoco.mj_resetData(model, data)
    # Deterministic initial angle per seed, away from limits.
    data.qpos[0] = 0.15 * np.sin(seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    mid_index = n_steps // 2
    mid_snap = None
    for step, tau in enumerate(torques):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        residual = generalized_force_residual(
            model, data, nominal_passive=nominal, include_constraint=True
        )
        records["qpos"].append(np.array(data.qpos, copy=True))
        records["qvel"].append(np.array(data.qvel, copy=True))
        records["qacc"].append(np.array(data.qacc, copy=True))
        records["tau_applied"].append(np.array(data.qfrc_applied, copy=True))
        records["qfrc_bias"].append(get_bias_force(data))
        records["qfrc_constraint"].append(get_constraint_force(data))
        records["qfrc_passive_truth"].append(get_truth_passive_force(data))
        records["residual"].append(residual)
        records["known_control"].append(known_control_force(data))
        if step == mid_index:
            mid_snap = _snapshot(model, data)

    packed = {key: np.stack(vals, axis=0) for key, vals in records.items()}
    nrmse = residual_nrmse(packed["residual"], packed["tau_applied"])
    close = bool(nrmse < config.nrmse_max)

    # Snapshot restore + deterministic replay checks (infrastructure plumbing).
    assert mid_snap is not None
    _restore(model, data, mid_snap)
    restore_err = float(
        np.max(
            np.abs(np.asarray(data.qpos) - mid_snap["qpos"])
            + np.abs(np.asarray(data.qvel) - mid_snap["qvel"])
        )
    )
    replay_a = []
    replay_b = []
    for tau in torques[mid_index : mid_index + 32]:
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        replay_a.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    _restore(model, data, mid_snap)
    for tau in torques[mid_index : mid_index + 32]:
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        replay_b.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    replay_err = float(np.max(np.abs(np.stack(replay_a) - np.stack(replay_b))))

    # Horizon plumbing markers (H32 blind-eval only).
    horizon_marks = {
        "acceptance_horizons": list(ACCEPTANCE_HORIZONS),
        "evaluation_horizons": list(HORIZONS),
        "h32_visible_to_acceptance": False,
        "passivity_sample": float(
            np.mean(packed["residual"][:, 0] * packed["qvel"][:, 0])
        ),
    }

    return {
        "seed": seed,
        "trajectory": trajectory,
        "nrmse": nrmse,
        "close": close,
        "restore_error": restore_err,
        "replay_error": replay_err,
        "dof_index": 0,
        "n_steps": n_steps,
        "timestep": config.timestep,
        "xml_sha256": _xml_sha256(config.timestep),
        "mass_matrix_shape": list(get_mass_matrix(model, data).shape),
        "horizon_plumbing": horizon_marks,
        "arrays": packed,
    }


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        meta_grp = handle.create_group("metadata")
        meta_grp.attrs["json"] = json.dumps(meta, sort_keys=True)
        raw = handle.create_group("raw_truth")
        learner = handle.create_group("learner_visible")
        for key in (
            "qpos",
            "qvel",
            "qacc",
            "qfrc_bias",
            "qfrc_constraint",
            "qfrc_passive_truth",
            "tau_applied",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        for key in ("qpos", "qvel", "qacc", "known_control", "residual"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        # Explicitly document leakage policy.
        learner.attrs["excludes_truth_qfrc_passive"] = True
        decision = handle.create_group("decision")
        decision.attrs["stage"] = "MJ0_infrastructure"
        evaluation = handle.create_group("evaluation")
        evaluation.create_dataset("residual", data=arrays["residual"], compression="gzip")
        evaluation.attrs["nrmse"] = cell["nrmse"]
        evaluation.attrs["close"] = cell["close"]


def all_cells_individually_close(rows: list[dict[str, Any]]) -> bool:
    required = {
        (seed, traj) for seed in MJ0_SEEDS for traj in MJ0_TRAJECTORIES
    }
    observed = {(row["seed"], row["trajectory"]) for row in rows}
    return observed == required and all(bool(row["close"]) for row in rows)


def run_r1_mj0(
    output: str | Path,
    *,
    config: R1MJ0Config | None = None,
) -> dict[str, Any]:
    cfg = config or R1MJ0Config()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for trajectory in cfg.trajectories:
            cell = _rollout_cell(cfg, seed=seed, trajectory=trajectory)
            rel = f"seed_{seed}/{trajectory}.hdf5"
            path = root / rel
            _write_cell_h5(path, cell)
            rows.append(
                {
                    "seed": seed,
                    "trajectory": trajectory,
                    "path": str(path),
                    "nrmse": cell["nrmse"],
                    "close": cell["close"],
                    "restore_error": cell["restore_error"],
                    "replay_error": cell["replay_error"],
                    "n_steps": cell["n_steps"],
                }
            )

    passed = all_cells_individually_close(rows)
    mujoco = import_mujoco()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")
    summary = {
        "stage": "R1-MJ0",
        "scope": "native MuJoCo single-hinge force-space closure",
        "scientific_result": False,
        "infrastructure_gate": True,
        "packages": {
            "mujoco": mujoco_version,
            "physics_only_import": bool(getattr(mujoco, "physics_only", False)),
        },
        "config": asdict(cfg),
        "xml_sha256": _xml_sha256(cfg.timestep),
        "nrmse_max": cfg.nrmse_max,
        "cells": rows,
        "cell_count": len(rows),
        "all_cells_individually_close": passed,
        "mj0_passed": passed,
        "unlocks_rs0": passed,
        "stage_status": stage_status(),
        "output": str(root.resolve()),
    }
    if passed:
        summary["stage_status"]["R1-RS0"] = {
            "locked_until_MJ0_passes": False,
            "unlocked": True,
        }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", summary["stage_status"])
    return summary
