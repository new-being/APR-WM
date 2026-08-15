"""R1-RS0: robosuite Door Mode-A C0 closure.

Locked until R1-MJ0 passes.  Robot remains present but inactive; torque is
applied directly to the door hinge.  Does not unlock RS1 revision science.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import (
    NominalPassiveParams,
    generalized_force_residual,
    get_bias_force,
    get_constraint_force,
    get_truth_passive_force,
    known_control_force,
    residual_nrmse,
)
from .r1_mj import stage_status


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


RS0_SEEDS = (8301, 8311, 8321)
RS0_PROFILES = ("P0", "P1", "P2")
RS0_DURATION_S = 10.0
RS0_NRMSE_MAX = 1.0e-3
RS0_FALSE_REVISION_MAX = 0.01
RS0_H32_VALIDITY_MIN = 0.95
# Door XML defaults frictionloss=0; use an explicit adequate nominal so P1/P2
# actually change passive physics rather than only damping.
RS0_NOMINAL_FRICTION = 0.1
RS0_NOMINAL_DAMPING = 0.1


@dataclass(frozen=True)
class PhysicsProfile:
    name: str
    friction_scale: float
    damping_scale: float


PROFILES = {
    "P0": PhysicsProfile("P0", 1.0, 1.0),
    "P1": PhysicsProfile("P1", 0.5, 0.5),
    "P2": PhysicsProfile("P2", 2.0, 2.0),
}


@dataclass(frozen=True)
class R1RS0Config:
    seeds: tuple[int, ...] = RS0_SEEDS
    profiles: tuple[str, ...] = RS0_PROFILES
    duration_s: float = RS0_DURATION_S
    nrmse_max: float = RS0_NRMSE_MAX
    false_revision_max: float = RS0_FALSE_REVISION_MAX
    h32_validity_min: float = RS0_H32_VALIDITY_MIN
    torque_amplitude: float = 0.12
    torque_freq_hz: float = 0.25
    use_latch: bool = False
    nominal_friction: float = RS0_NOMINAL_FRICTION
    nominal_damping: float = RS0_NOMINAL_DAMPING
    disable_fluid: bool = True
    freeze_robot: bool = True
    joint_limit_margin: float = 0.05
    min_modeled_fraction: float = 0.50


def _require_mj0_unlock(mj0_summary: Path | None) -> dict[str, Any]:
    if mj0_summary is None or not Path(mj0_summary).is_file():
        raise RuntimeError(
            "R1-RS0 is locked until R1-MJ0 passes; provide --mj0-summary"
        )
    payload = json.loads(Path(mj0_summary).read_text(encoding="utf-8"))
    if not payload.get("mj0_passed"):
        raise RuntimeError("R1-RS0 remains locked: mj0_passed is false")
    return payload


def _find_hinge_dof(model: Any, preferred_names: tuple[str, ...] = (
    "hinge",
    "door_hinge",
    "Door_hinge",
    "door_joint",
)) -> int:
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco()

    names = []
    for jnt_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
        names.append(name)
        adr = int(model.jnt_dofadr[jnt_id])
        if any(key.lower() in name.lower() for key in preferred_names):
            return adr
    # Fallback: first hinge joint.
    for jnt_id in range(model.njnt):
        if int(model.jnt_type[jnt_id]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            return int(model.jnt_dofadr[jnt_id])
    raise RuntimeError(f"No hinge DOF found; joints={names}")


def _joint_id_for_dof(model: Any, dof: int) -> int:
    for jnt_id in range(model.njnt):
        if int(model.jnt_dofadr[jnt_id]) == int(dof):
            return int(jnt_id)
    raise RuntimeError(f"No joint maps to dof {dof}")


def _read_joint_passive(model: Any, joint_id: int) -> tuple[float, float]:
    damping = float(model.dof_damping[model.jnt_dofadr[joint_id]])
    friction = float(model.dof_frictionloss[model.jnt_dofadr[joint_id]])
    return friction, damping


def _set_joint_passive(
    model: Any, joint_id: int, *, friction: float, damping: float
) -> None:
    dof = int(model.jnt_dofadr[joint_id])
    model.dof_damping[dof] = float(damping)
    model.dof_frictionloss[dof] = float(friction)


def _make_door_env(*, seed: int, use_latch: bool) -> Any:
    from .mujoco_physics import import_mujoco

    # Must register MuJoCo 3.11 qM/mj_fullM shims before robosuite constructs OSC.
    import_mujoco(register=True)
    import robosuite as suite

    # Robot is present but unused: Mode-A applies hinge torque directly.
    env = suite.make(
        "Door",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True,
        use_latch=use_latch,
        seed=seed,
    )
    env.reset()
    return env


def _mujoco_handles(env: Any) -> tuple[Any, Any]:
    sim = getattr(env, "sim", None)
    if sim is None:
        raise TypeError("robosuite env does not expose .sim")
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        # Some wrappers expose mj_model / mj_data.
        model = getattr(sim, "mj_model", model)
        data = getattr(sim, "mj_data", data)
    if model is None or data is None:
        raise TypeError("Could not locate MuJoCo model/data on robosuite env")
    # robosuite binding wrappers store the native objects under _model/_data.
    model = getattr(model, "_model", model)
    data = getattr(data, "_data", data)
    return model, data


def _rollout_rs0_cell(
    config: R1RS0Config,
    *,
    seed: int,
    profile_name: str,
) -> dict[str, Any]:
    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco(register=True)

    profile = PROFILES[profile_name]
    env = _make_door_env(seed=seed, use_latch=config.use_latch)
    try:
        model, data = _mujoco_handles(env)
        # Mode-A C0 is about hinge friction/damping closure, not fluid drag.
        if config.disable_fluid:
            model.opt.viscosity = 0.0
            model.opt.density = 0.0
        hinge_dof = _find_hinge_dof(model)
        joint_id = _joint_id_for_dof(model, hinge_dof)
        friction = float(config.nominal_friction) * profile.friction_scale
        damping = float(config.nominal_damping) * profile.damping_scale
        _set_joint_passive(model, joint_id, friction=friction, damping=damping)
        # MuJoCo places frictionloss inside constraint forces (qfrc_constraint),
        # while viscous damping appears in qfrc_passive.  Learner nominal therefore
        # only carries damping; frictionloss is accounted via include_constraint.
        nominal = NominalPassiveParams(damping=damping, frictionloss=0.0)

        # Simulation-rate logging, not control-rate only.
        dt = float(model.opt.timestep)
        n_steps = int(round(config.duration_s / dt))
        t = np.arange(n_steps, dtype=np.float64) * dt
        torques = config.torque_amplitude * np.sin(
            2.0 * np.pi * config.torque_freq_hz * t
        )

        records: dict[str, list[np.ndarray]] = {
            "q": [],
            "qvel": [],
            "qacc": [],
            "tau_applied": [],
            "known_control": [],
            "residual": [],
            "qfrc_bias": [],
            "qfrc_constraint": [],
            "qfrc_passive_truth": [],
            "validity": [],
        }

        # Zero robot actuators; only hinge generalized torque is applied.
        if hasattr(data, "ctrl"):
            data.ctrl[:] = 0.0
        # Start mid-range so Mode-A excitation stays inside modeled support.
        hinge_qpos_adr = int(model.jnt_qposadr[joint_id])
        lo, hi = [float(x) for x in model.jnt_range[joint_id]]
        if hi > lo:
            data.qpos[hinge_qpos_adr] = 0.5 * (lo + hi)
            data.qvel[hinge_dof] = 0.0
        # Kinematic freeze implements "robot fixed / inactive" without contact
        # excitation from a collapsing arm under gravity. Only non-hinge DOFs
        # are frozen; the door hinge remains free for Mode-A excitation.
        robot_qpos0 = np.array(data.qpos, dtype=np.float64, copy=True)
        robot_qpos_idx = [i for i in range(model.nq) if i != hinge_qpos_adr]
        robot_dof_idx = [i for i in range(model.nv) if i != hinge_dof]
        mujoco.mj_forward(model, data)

        for tau in torques:
            if config.freeze_robot:
                data.qpos[robot_qpos_idx] = robot_qpos0[robot_qpos_idx]
                data.qvel[robot_dof_idx] = 0.0
                mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[hinge_dof] = float(tau)
            mujoco.mj_step(model, data)
            residual_full = generalized_force_residual(
                model,
                data,
                nominal_passive=nominal,
                include_constraint=True,
                dof_index=hinge_dof,
            )
            # Near joint limits → boundary; otherwise modeled.
            q = float(data.qpos[hinge_qpos_adr])
            margin = min(q - lo, hi - q) if hi > lo else 1.0
            validity = 1.0 if margin > config.joint_limit_margin else 0.5
            records["q"].append(np.array([q], dtype=np.float64))
            records["qvel"].append(np.array([data.qvel[hinge_dof]], dtype=np.float64))
            records["qacc"].append(np.array([data.qacc[hinge_dof]], dtype=np.float64))
            records["tau_applied"].append(
                np.array([data.qfrc_applied[hinge_dof]], dtype=np.float64)
            )
            records["known_control"].append(
                np.array([known_control_force(data)[hinge_dof]], dtype=np.float64)
            )
            records["residual"].append(
                np.array([residual_full[hinge_dof]], dtype=np.float64)
            )
            records["qfrc_bias"].append(
                np.array([get_bias_force(data)[hinge_dof]], dtype=np.float64)
            )
            records["qfrc_constraint"].append(
                np.array([get_constraint_force(data)[hinge_dof]], dtype=np.float64)
            )
            records["qfrc_passive_truth"].append(
                np.array([get_truth_passive_force(data)[hinge_dof]], dtype=np.float64)
            )
            records["validity"].append(np.array([validity], dtype=np.float64))

        packed = {key: np.stack(vals, axis=0) for key, vals in records.items()}
        modeled_mask = packed["validity"][:, 0] >= 0.999
        if not bool(np.any(modeled_mask)):
            nrmse = float("inf")
        else:
            nrmse = residual_nrmse(
                packed["residual"][modeled_mask],
                packed["tau_applied"][modeled_mask],
            )
        h32 = min(32, packed["residual"].shape[0])
        h32_validity = float(np.mean(packed["validity"][-h32:, 0] >= 0.999))
        modeled_fraction = float(np.mean(modeled_mask))
        # C0 adequate: no revision should fire; false revision proxy = nrmse fail.
        false_revision = 0.0 if nrmse < config.nrmse_max else 1.0
        close = bool(
            nrmse < config.nrmse_max
            and false_revision <= config.false_revision_max
            and h32_validity >= config.h32_validity_min
            and modeled_fraction >= config.min_modeled_fraction
        )
        return {
            "seed": seed,
            "profile": profile_name,
            "hinge_dof": hinge_dof,
            "base_friction": float(config.nominal_friction),
            "base_damping": float(config.nominal_damping),
            "friction": friction,
            "damping": damping,
            "nrmse": nrmse,
            "false_revision_rate": false_revision,
            "h32_validity_coverage": h32_validity,
            "modeled_fraction": modeled_fraction,
            "close": close,
            "timestep": dt,
            "n_steps": n_steps,
            "disable_fluid": bool(config.disable_fluid),
            "freeze_robot": bool(config.freeze_robot),
            "learner_models_frictionloss": False,
            "frictionloss_via_constraint": True,
            "arrays": packed,
        }
    finally:
        env.close()


def _write_cell_h5(path: Path, cell: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = cell["arrays"]
    meta = {key: value for key, value in cell.items() if key != "arrays"}
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["json"] = json.dumps(meta, sort_keys=True)
        raw = handle.create_group("raw_truth")
        learner = handle.create_group("learner_visible")
        for key in (
            "q",
            "qvel",
            "qacc",
            "qfrc_bias",
            "qfrc_constraint",
            "qfrc_passive_truth",
            "tau_applied",
        ):
            raw.create_dataset(key, data=arrays[key], compression="gzip")
        for key in ("q", "qvel", "qacc", "known_control", "residual", "validity"):
            learner.create_dataset(key, data=arrays[key], compression="gzip")
        learner.attrs["excludes_truth_qfrc_passive"] = True
        handle.create_group("decision").attrs["stage"] = "RS0_C0"
        evaluation = handle.create_group("evaluation")
        evaluation.attrs["nrmse"] = cell["nrmse"]
        evaluation.attrs["close"] = cell["close"]
        evaluation.attrs["h32_validity_coverage"] = cell["h32_validity_coverage"]


def all_cells_individually_close(rows: list[dict[str, Any]]) -> bool:
    required = {(seed, profile) for seed in RS0_SEEDS for profile in RS0_PROFILES}
    observed = {(row["seed"], row["profile"]) for row in rows}
    return observed == required and all(bool(row["close"]) for row in rows)


def run_r1_rs0(
    output: str | Path,
    *,
    mj0_summary: str | Path,
    config: R1RS0Config | None = None,
) -> dict[str, Any]:
    cfg = config or R1RS0Config()
    mj0 = _require_mj0_unlock(Path(mj0_summary))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    from .mujoco_physics import import_mujoco

    mujoco = import_mujoco(register=True)
    import robosuite

    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for profile in cfg.profiles:
            cell = _rollout_rs0_cell(cfg, seed=seed, profile_name=profile)
            rel = f"profile_{profile}/seed_{seed}.hdf5"
            path = root / rel
            _write_cell_h5(path, cell)
            rows.append(
                {
                    "seed": seed,
                    "profile": profile,
                    "path": str(path),
                    "nrmse": cell["nrmse"],
                    "close": cell["close"],
                    "false_revision_rate": cell["false_revision_rate"],
                    "h32_validity_coverage": cell["h32_validity_coverage"],
                    "hinge_dof": cell["hinge_dof"],
                    "friction": cell["friction"],
                    "damping": cell["damping"],
                }
            )

    passed = all_cells_individually_close(rows)
    status = stage_status()
    status["R1-MJ0"] = {"unlocked": True, "passed": True}
    status["R1-RS0"] = {
        "locked_until_MJ0_passes": False,
        "unlocked": True,
        "passed": passed,
    }
    if passed:
        status["R1-RS1"] = {"locked": False, "unlocked": True}
    summary = {
        "stage": "R1-RS0",
        "scope": "robosuite Door Mode-A C0 closure across physics profiles",
        "scientific_result": False,
        "packages": {
            "robosuite": getattr(robosuite, "__version__", "unknown"),
            "mujoco": getattr(mujoco, "__version__", "unknown"),
        },
        "mj0_summary": str(Path(mj0_summary).resolve()),
        "mj0_passed": bool(mj0.get("mj0_passed")),
        "config": asdict(cfg),
        "cells": rows,
        "cell_count": len(rows),
        "all_cells_individually_close": passed,
        "rs0_passed": passed,
        "unlocks_rs1": passed,
        "stage_status": status,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "stage_status.json", status)
    return summary


# Backwards-compatible thin wrapper name used by early README draft.
def run_r1_rs_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError(
        "r1-rs-collect is retired; use r1-mj0 then r1-rs0 under ordered gates"
    )
