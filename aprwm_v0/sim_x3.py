"""SIM-X3: persistent validity lifecycle from X2 information channel.

Three-hypothesis analytic Bayes. Canonical license start. No detector,
CUSUM, NetVoI, or R10 unlock.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .mujoco_force import generalized_force_residual
from .mujoco_physics import import_mujoco
from .sim_x2 import SIMX2_ACTIONS
from .simx_plant import (
    HOST_PLANT_ID,
    SIMX_DAMPING,
    assert_x1_plant,
    learner_nominal_params,
    load_simx_hinge,
)

PREREG_PATH = "REPORT/REG/SIMX/SIMX3_PREREG.md"
SCHEMA_ID = "aprwm.sim_x3.lifecycle.v1"
EPS = 1.0e-12

HYPOTHESES = (0.05, 0.10, 0.15)
B0 = SIMX_DAMPING
CANONICAL_W = np.array([0.025, 0.95, 0.025], dtype=np.float64)
LICENSE_THRESH = 0.5
PHASES = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
ACTIONS = ("info", "cons", "high_f")
PLANT_CONDITIONS = (0.10, 0.05, 0.15)  # post-shift true damping


@dataclass(frozen=True)
class SIMX3Config:
    actions: tuple[str, ...] = ACTIONS
    plant_conditions: tuple[float, ...] = PLANT_CONDITIONS
    phases: tuple[float, ...] = PHASES
    n_noise_seeds: int = 100
    noise_seed0: int = 20_000
    physics_seed: int = 9101
    pre_roll_s: float = 2.0
    lifecycle_s: float = 4.0
    timestep: float = 0.002
    sigma_obs: float = 0.01
    learner_damping: float = B0
    license_thresh: float = LICENSE_THRESH
    g1_false_revoke_max: float = 0.01
    g2_info_revoke_min: float = 0.95
    g4_highf_persist_min: float = 0.90
    oracle_tol: float = 1.0e-4


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def torque_with_phase(
    action: str,
    *,
    n_steps: int,
    dt: float,
    phase: float,
) -> np.ndarray:
    spec = SIMX2_ACTIONS[action]
    t = np.arange(n_steps, dtype=np.float64) * dt
    return float(spec["amplitude"]) * np.sin(
        2.0 * np.pi * float(spec["freq_hz"]) * t + float(phase)
    )


def simulate_plant_block(
    config: SIMX3Config,
    *,
    action: str,
    phase: float,
    post_damping: float,
) -> dict[str, np.ndarray]:
    """Pre-roll at b=0.10 (no belief), then lifecycle under post_damping."""

    dt = config.timestep
    n_pre = int(round(config.pre_roll_s / dt))
    n_life = int(round(config.lifecycle_s / dt))
    torques = torque_with_phase(
        action, n_steps=n_pre + n_life, dt=dt, phase=phase
    )
    # Continuous rollout: start at 0.10, then switch dof_damping.
    mujoco = import_mujoco()
    model, data = load_simx_hinge(dt, true_damping=0.10)
    assert_x1_plant(model, true_damping=0.10)
    learner = learner_nominal_params()

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.15 * np.sin(config.physics_seed * 0.001)
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    def _step(tau: float, damp: float) -> tuple[float, float, float]:
        model.dof_damping[0] = float(damp)
        qvel_pre = float(data.qvel[0])
        data.qfrc_applied[:] = 0.0
        data.qfrc_actuator[:] = 0.0
        data.ctrl[:] = 0.0
        data.qfrc_applied[0] = float(tau)
        mujoco.mj_step(model, data)
        residual = float(
            generalized_force_residual(
                model,
                data,
                nominal_passive=learner,
                include_constraint=True,
                dof_index=0,
                qvel_for_passive=np.array([qvel_pre], dtype=np.float64),
            )[0]
        )
        delta_b = float(damp) - float(learner.damping)
        return qvel_pre, residual, -delta_b * qvel_pre

    for i in range(n_pre):
        _step(float(torques[i]), 0.10)

    qvel = np.zeros(n_life, dtype=np.float64)
    residual = np.zeros(n_life, dtype=np.float64)
    r_oracle = np.zeros(n_life, dtype=np.float64)
    for j in range(n_life):
        qv, r, ro = _step(float(torques[n_pre + j]), float(post_damping))
        qvel[j] = qv
        residual[j] = r
        r_oracle[j] = ro

    return {"qvel": qvel, "residual": residual, "r_oracle": r_oracle}


def bayes_lifecycle(
    *,
    qvel: np.ndarray,
    residual_truth: np.ndarray,
    sigma_obs: float,
    noise_seed: int,
    w0: np.ndarray = CANONICAL_W,
    hypotheses: tuple[float, ...] = HYPOTHESES,
    b0: float = B0,
    license_thresh: float = LICENSE_THRESH,
    dt: float = 0.002,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(noise_seed))
    n = int(qvel.size)
    noise = rng.normal(0.0, float(sigma_obs), size=n)
    y = residual_truth + noise

    log_w = np.log(np.asarray(w0, dtype=np.float64) + EPS)
    b_traj = np.empty(n, dtype=np.float64)
    # Store only terminal weights in aggregate; keep optional mid samples light.
    for t in range(n):
        qd = float(qvel[t])
        yt = float(y[t])
        for k, h in enumerate(hypotheses):
            mu = -(float(h) - float(b0)) * qd
            # log N(yt; mu, σ²) ∝ -0.5 (yt-mu)²/σ²
            log_w[k] += -0.5 * ((yt - mu) / float(sigma_obs)) ** 2
        log_w -= np.logaddexp.reduce(log_w)
        w = np.exp(log_w)
        b_traj[t] = float(w[1])  # H=0.10

    licensed = b_traj >= float(license_thresh)
    s_stale = float(np.mean(licensed))
    below = np.where(~licensed)[0]
    if below.size == 0:
        t_revoke = None
        censored = True
    else:
        t_revoke = float(below[0] * dt)
        censored = False
    f_revoke = bool(below.size > 0)
    w_T = np.exp(log_w)
    return {
        "b_traj": b_traj,
        "y": y,
        "s_stale": s_stale,
        "t_revoke": t_revoke,
        "censored": censored,
        "f_revoke": f_revoke,
        "b_T": float(b_traj[-1]),
        "w_T": w_T.tolist(),
        "licensed_frac": s_stale,
    }


def _pattern(summary_gates: dict[str, Any], metrics: dict[str, Any]) -> str:
    if not summary_gates["g1_valid_retention"]:
        return "false_revoke"
    if summary_gates["g2_informative_revocation"] and summary_gates["g4_near_blind_persistence"]:
        if summary_gates["g3_lifecycle_ordering"]:
            return "lifecycle_blindness"
    # high_f also revokes often
    if summary_gates["g2_informative_revocation"] and not summary_gates[
        "g4_near_blind_persistence"
    ]:
        return "accumulation_recovers"
    if not summary_gates["g1_valid_retention"]:
        return "false_revoke"
    return "mixed_or_inconclusive"


def run_sim_x3(
    output: str | Path,
    *,
    config: SIMX3Config | None = None,
    x2_summary: str | Path | None = "runs/sim_x2/formal/summary.json",
) -> dict[str, Any]:
    cfg = config or SIMX3Config()
    if abs(cfg.sigma_obs - 0.01) > 1.0e-12:
        raise RuntimeError("SIM-X3 forbids retuning sigma_obs away from 0.01")
    if abs(cfg.license_thresh - 0.5) > 1.0e-12:
        raise RuntimeError("SIM-X3 forbids retuning license threshold away from 0.5")
    if abs(cfg.learner_damping - B0) > 1.0e-12:
        raise RuntimeError("SIM-X3 learner_damping must remain frozen at 0.10")

    root = Path(output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("SIM-X3 must not write under runs/r10_c0/")
    if x2_summary is not None:
        path = Path(x2_summary)
        if not path.is_file():
            raise RuntimeError(f"SIM-X3 locked until SIM-X2 summary exists ({path})")
        x2 = json.loads(path.read_text(encoding="utf-8"))
        if not bool(x2.get("sim_x2_passed")):
            raise RuntimeError("SIM-X3 locked until sim_x2_passed=true")
        if x2.get("host_plant_id") != HOST_PLANT_ID:
            raise RuntimeError("SIM-X3 requires X2 host_plant_id=simx_hinge.v1")
        if abs(float(x2.get("sigma_obs", -1.0)) - 0.01) > 1.0e-12:
            raise RuntimeError("SIM-X3 requires X2 sigma_obs=0.01 carryover")

    root.mkdir(parents=True, exist_ok=True)

    # Cache plant trajectories: (action, phase, post_damping) -> arrays
    plant_cache: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}
    e_oracle_vals: list[float] = []
    for action in cfg.actions:
        for phase in cfg.phases:
            for post_b in cfg.plant_conditions:
                key = (action, float(phase), float(post_b))
                traj = simulate_plant_block(
                    cfg, action=action, phase=float(phase), post_damping=float(post_b)
                )
                plant_cache[key] = traj
                if abs(float(post_b) - 0.10) > 1.0e-12:
                    r = traj["residual"]
                    ro = traj["r_oracle"]
                    rmse = float(np.sqrt(np.mean(np.square(r - ro))))
                    rms = float(np.sqrt(np.mean(np.square(ro))))
                    e_oracle_vals.append(rmse / (rms + 1.0e-8))

    g0 = bool(max(e_oracle_vals) < cfg.oracle_tol) if e_oracle_vals else False

    rows: list[dict[str, Any]] = []
    # Optional compact HDF5 of a few exemplars only; full 3600 as JSONL metrics.
    metrics_path = root / "cells.jsonl"
    exemplars_dir = root / "exemplars"
    exemplars_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as sink:
        for action in cfg.actions:
            for phase in cfg.phases:
                for post_b in cfg.plant_conditions:
                    traj = plant_cache[(action, float(phase), float(post_b))]
                    for n_i in range(cfg.n_noise_seeds):
                        noise_seed = cfg.noise_seed0 + n_i
                        life = bayes_lifecycle(
                            qvel=traj["qvel"],
                            residual_truth=traj["residual"],
                            sigma_obs=cfg.sigma_obs,
                            noise_seed=noise_seed,
                            dt=cfg.timestep,
                            license_thresh=cfg.license_thresh,
                        )
                        row = {
                            "action": action,
                            "phase": float(phase),
                            "plant_condition": float(post_b),
                            "invalid": abs(float(post_b) - 0.10) > 1.0e-12,
                            "noise_seed": int(noise_seed),
                            "s_stale": life["s_stale"],
                            "t_revoke": life["t_revoke"],
                            "censored": life["censored"],
                            "f_revoke": life["f_revoke"],
                            "b_T": life["b_T"],
                            "w_T": life["w_T"],
                        }
                        sink.write(json.dumps(row) + "\n")
                        rows.append(row)

                        # Save a few exemplars for inspection.
                        if (
                            n_i == 0
                            and abs(float(phase)) < 1.0e-12
                            and action in ("info", "cons", "high_f")
                        ):
                            h5 = (
                                exemplars_dir
                                / f"{action}_b{post_b:.2f}_phase0_noise0.h5"
                            )
                            with h5py.File(h5, "w") as handle:
                                handle.attrs["schema"] = SCHEMA_ID
                                handle.attrs["action"] = action
                                handle.attrs["plant_condition"] = float(post_b)
                                handle.create_dataset("qvel", data=traj["qvel"])
                                handle.create_dataset("residual", data=traj["residual"])
                                handle.create_dataset("b_traj", data=life["b_traj"])
                                handle.create_dataset("y", data=life["y"])

    def _subset(
        *,
        action: str | None = None,
        invalid: bool | None = None,
        plant: float | None = None,
    ) -> list[dict[str, Any]]:
        out = rows
        if action is not None:
            out = [r for r in out if r["action"] == action]
        if invalid is not None:
            out = [r for r in out if bool(r["invalid"]) == invalid]
        if plant is not None:
            out = [r for r in out if abs(float(r["plant_condition"]) - plant) < 1.0e-12]
        return out

    stay = _subset(invalid=False)
    p_false = float(np.mean([r["f_revoke"] for r in stay])) if stay else 1.0
    g1 = bool(p_false <= cfg.g1_false_revoke_max)

    info_inv = _subset(action="info", invalid=True)
    p_info_rev = float(
        np.mean([not r["censored"] for r in info_inv])
    ) if info_inv else 0.0
    g2 = bool(p_info_rev >= cfg.g2_info_revoke_min)

    def mean_stale(action: str, plant: float | None = None) -> float:
        sub = _subset(action=action, invalid=True, plant=plant)
        return float(np.mean([r["s_stale"] for r in sub])) if sub else float("nan")

    ordering_ok = True
    stale_by: dict[str, Any] = {"aggregate": {}, "by_shift": {}}
    for label, plant in (("aggregate", None), ("shift_0.05", 0.05), ("shift_0.15", 0.15)):
        s_info = mean_stale("info", plant)
        s_cons = mean_stale("cons", plant)
        s_high = mean_stale("high_f", plant)
        bucket = stale_by["aggregate"] if plant is None else stale_by["by_shift"]
        key = "all_invalid" if plant is None else f"b{plant:.2f}"
        bucket[key] = {"info": s_info, "cons": s_cons, "high_f": s_high}
        if not (s_info < s_cons < s_high):
            ordering_ok = False
    g3 = bool(ordering_ok)

    high_inv = _subset(action="high_f", invalid=True)
    p_high_persist = float(
        np.mean([r["censored"] for r in high_inv])
    ) if high_inv else 0.0
    g4 = bool(p_high_persist >= cfg.g4_highf_persist_min)

    gates = {
        "g0_x2_carryover": g0,
        "g1_valid_retention": g1,
        "g2_informative_revocation": g2,
        "g3_lifecycle_ordering": g3,
        "g4_near_blind_persistence": g4,
    }
    passed = all(gates.values())
    pattern = _pattern(gates, stale_by)

    # Median revoke times among uncensored invalid cells (diagnostic).
    def median_revoke(action: str) -> float | None:
        times = [
            r["t_revoke"]
            for r in _subset(action=action, invalid=True)
            if r["t_revoke"] is not None
        ]
        return float(np.median(times)) if times else None

    mujoco = import_mujoco()
    try:
        from importlib import metadata

        mujoco_version = metadata.version("mujoco")
    except Exception:
        mujoco_version = getattr(mujoco, "__version__", "unknown")

    summary = {
        "stage": "SIM-X3",
        "schema": SCHEMA_ID,
        "source": "simulator",
        "host_plant_id": HOST_PLANT_ID,
        "prereg": PREREG_PATH,
        "scientific_claim": "X2 channel ordering → persistent belief lifecycle ordering",
        "real_physics": False,
        "unlocks_r10_c0": False,
        "sigma_obs": cfg.sigma_obs,
        "sigma_obs_kind": "synthetic_observation_resolution_surrogate",
        "canonical_w0": CANONICAL_W.tolist(),
        "license_thresh": cfg.license_thresh,
        "packages": {"mujoco": mujoco_version},
        "config": asdict(cfg),
        "cell_count": len(rows),
        "max_e_oracle": max(e_oracle_vals) if e_oracle_vals else None,
        "p_false_revoke_stay": p_false,
        "p_revoke_info_invalid": p_info_rev,
        "p_persist_highf_invalid": p_high_persist,
        "s_stale": stale_by,
        "median_t_revoke_uncensored": {
            a: median_revoke(a) for a in cfg.actions
        },
        **{f"gate_{k}": v for k, v in gates.items()},
        "g0_x2_carryover": g0,
        "g1_valid_retention": g1,
        "g2_informative_revocation": g2,
        "g3_lifecycle_ordering": g3,
        "g4_near_blind_persistence": g4,
        "sim_x3_passed": passed,
        "pattern": pattern,
        "claims_if_pattern_A": {
            "cross_engine_lifecycle_robustness": pattern == "lifecycle_blindness",
            "real_physics": False,
        },
        "cells_jsonl": str(metrics_path.resolve()),
        "output": str(root.resolve()),
    }
    # flatten gate keys already set
    _write_json(root / "summary.json", summary)
    return summary
