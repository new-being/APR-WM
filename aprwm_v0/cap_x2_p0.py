"""CAP-X2-P0: oracle-only planning harness lock (before any capacity sweep).

Chooses the smallest (cand, iter) with oracle success >= 0.80 on reachable
targets, or disables planning for X2. Does not inspect PureNN/Hybrid curves.
Does not unlock R10.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .capx_arm3_plant import (
    HOST_PLANT_ID,
    N_DOF,
    SceneTheta,
    apply_scene_theta,
    load_capx_arm3,
    sample_scene_theta,
)
from .capx_residual import (
    RESIDUAL_FAMILY_ID,
    SceneResidualCoeffs,
    sample_residual_coeffs,
    tau_perp,
)
from .mujoco_force import get_bias_force, get_mass_matrix
from .mujoco_physics import import_mujoco
from .capx_arm3_plant import per_dof_passive_force

PREREG_PATH = "REPORT/REG/CAPX/CAPX2_PREREG.md"
SCHEMA_ID = "aprwm.cap_x2_p0.harness.v1"

PLAN_HORIZON_S = 1.0
MODEL_DT = 0.01
ACTION_KNOTS = 20
ELITE_FRAC = 0.10
EPS_Q = 0.15
LAMBDA_V = 0.05
LAMBDA_U = 0.01
ORACLE_SUCCESS_MIN = 0.80
# Frozen P0 budget grid (ascending); pick smallest that passes.
BUDGET_GRID: tuple[tuple[int, int], ...] = (
    (256, 5),
    (512, 5),
    (512, 8),
    (1024, 8),
)


@dataclass(frozen=True)
class CAPX2P0Config:
    output: str = "runs/cap_x2/p0"
    # P0 locks at rho=0 plant (prereg default).
    rho: float = 0.0
    n_tasks: int = 24
    n_scenes: int = 12
    scene_seed0: int = 91_000
    task_seed: int = 202_608
    u_abs_max: float = 1.5
    physics_timestep: float = 0.002


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def oracle_qdd(
    model: Any,
    data: Any,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    u: np.ndarray,
    theta: SceneTheta,
    coeffs: SceneResidualCoeffs,
    gamma: float,
    theta_cache: dict[str, Any] | None = None,
) -> np.ndarray:
    """True plant acceleration: library physics + tau_perp."""

    mujoco = import_mujoco()
    fp = theta.fingerprint()
    if theta_cache is None or theta_cache.get("fp") != fp:
        apply_scene_theta(model, data, theta)
        if theta_cache is not None:
            theta_cache["fp"] = fp
            theta_cache["theta"] = theta
    else:
        theta = theta_cache["theta"]
    data.qpos[:] = np.asarray(q, dtype=np.float64)
    data.qvel[:] = np.asarray(qd, dtype=np.float64)
    if model.nu:
        data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    mass = get_mass_matrix(model, data)
    bias = get_bias_force(data)
    passive = per_dof_passive_force(qd, theta)
    tau_p = tau_perp(q, qd, coeffs, gamma=gamma)
    rhs = np.asarray(u, dtype=np.float64) + passive - bias + tau_p
    return np.linalg.solve(mass, rhs)


def _rollout_to_target(
    predict: Callable,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    theta: np.ndarray,
    u_seq: np.ndarray,
) -> np.ndarray:
    q = q0.copy()
    qd = qd0.copy()
    for t in range(u_seq.shape[0]):
        qdd = np.asarray(predict(q, qd, u_seq[t], theta), dtype=np.float64)
        q = q + qd * MODEL_DT
        qd = qd + qdd * MODEL_DT
    return q


def cem_plan_oracle(
    predict: Callable,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    theta: np.ndarray,
    q_star: np.ndarray,
    q_abs_max: np.ndarray,
    u_abs_max: float,
    n_cand: int,
    n_iters: int,
    rng: np.random.Generator,
) -> tuple[bool, float, float]:
    horizon_steps = int(round(PLAN_HORIZON_S / MODEL_DT))
    knots = ACTION_KNOTS
    seg = max(1, horizon_steps // knots)
    mean = np.zeros((knots, N_DOF), dtype=np.float64)
    std = np.full((knots, N_DOF), 0.5 * u_abs_max, dtype=np.float64)
    n_elite = max(1, int(round(ELITE_FRAC * n_cand)))
    best_ok = False
    best_dist = float("inf")
    t0 = time.perf_counter()
    for _ in range(n_iters):
        samples = rng.normal(mean[None, :, :], std[None, :, :], size=(n_cand, knots, N_DOF))
        samples = np.clip(samples, -u_abs_max, u_abs_max)
        costs = np.zeros(n_cand, dtype=np.float64)
        dists = np.zeros(n_cand, dtype=np.float64)
        viols = np.zeros(n_cand, dtype=bool)
        for c in range(n_cand):
            q = q0.copy()
            qd = qd0.copy()
            cost = 0.0
            violated = False
            for t in range(horizon_steps):
                u = samples[c, min(t // seg, knots - 1)]
                qdd = np.asarray(predict(q, qd, u, theta), dtype=np.float64)
                q = q + qd * MODEL_DT
                qd = qd + qdd * MODEL_DT
                if np.any(np.abs(q) > q_abs_max):
                    violated = True
                cost += float(
                    np.sum((q - q_star) ** 2)
                    + LAMBDA_V * np.sum(qd**2)
                    + LAMBDA_U * np.sum(u**2)
                )
            dist = float(np.max(np.abs(q - q_star)))
            costs[c] = cost + (1.0e3 if violated else 0.0)
            dists[c] = dist
            viols[c] = violated
        elite = np.argpartition(costs, n_elite - 1)[:n_elite]
        mean = samples[elite].mean(axis=0)
        std = samples[elite].std(axis=0) + 1.0e-3
        # Success / distance: best terminal among non-violating (not cost-argmin alone).
        for c in range(n_cand):
            if viols[c]:
                continue
            if dists[c] < best_dist:
                best_dist = float(dists[c])
                best_ok = best_dist < EPS_Q
    wall = time.perf_counter() - t0
    return best_ok, wall, best_dist


def _make_tasks(config: CAPX2P0Config) -> list[dict[str, Any]]:
    """Build reachable-target tasks on rho=0 oracle plant."""

    assert float(config.rho) == 0.0, "P0 default lock uses rho=0"
    rng = np.random.default_rng(config.task_seed)
    model, data = load_capx_arm3(config.physics_timestep)
    horizon_steps = int(round(PLAN_HORIZON_S / MODEL_DT))
    tasks: list[dict[str, Any]] = []
    for i in range(config.n_tasks):
        scene_seed = config.scene_seed0 + (i % config.n_scenes)
        scene_rng = np.random.default_rng(scene_seed)
        theta = sample_scene_theta(scene_rng)
        coeffs = sample_residual_coeffs(np.random.default_rng(scene_seed + 17_777))
        gamma = 0.0
        q0 = scene_rng.uniform(-0.6, 0.6, size=N_DOF)
        qd0 = scene_rng.uniform(-0.3, 0.3, size=N_DOF)
        # Oracle open-loop torque sequence within action bounds.
        u_oracle = rng.uniform(-config.u_abs_max, config.u_abs_max, size=(horizon_steps, N_DOF))
        # Smooth a bit: piecewise-constant knots.
        knots = ACTION_KNOTS
        seg = max(1, horizon_steps // knots)
        u_knots = rng.uniform(-config.u_abs_max, config.u_abs_max, size=(knots, N_DOF))
        for t in range(horizon_steps):
            u_oracle[t] = u_knots[min(t // seg, knots - 1)]

        cache: dict[str, Any] = {}

        def predict(q, qd, u, _theta_vec, _th=theta, _co=coeffs, _g=gamma, _c=cache):
            return oracle_qdd(
                model,
                data,
                q=q,
                qd=qd,
                u=u,
                theta=_th,
                coeffs=_co,
                gamma=_g,
                theta_cache=_c,
            )

        q_star = _rollout_to_target(
            predict, q0=q0, qd0=qd0, theta=theta.as_vector(), u_seq=u_oracle
        )
        q_abs_max = np.maximum(np.abs(q0), np.abs(q_star)) * 1.5 + 0.75
        tasks.append(
            {
                "q0": q0,
                "qd0": qd0,
                "q_star": q_star,
                "theta": theta.as_vector(),
                "theta_obj": theta,
                "coeffs": coeffs,
                "gamma": gamma,
                "q_abs_max": q_abs_max,
                "u_oracle": u_oracle,
            }
        )
    return tasks


def evaluate_budget(
    tasks: list[dict[str, Any]],
    *,
    n_cand: int,
    n_iters: int,
    u_abs_max: float,
    physics_timestep: float,
    seed: int,
) -> dict[str, Any]:
    model, data = load_capx_arm3(physics_timestep)
    rng = np.random.default_rng(seed)
    oks: list[bool] = []
    walls: list[float] = []
    dists: list[float] = []
    for task in tasks:
        cache: dict[str, Any] = {}
        th = task["theta_obj"]
        co = task["coeffs"]
        gam = float(task["gamma"])

        def predict(q, qd, u, _theta_vec, _th=th, _co=co, _g=gam, _c=cache):
            return oracle_qdd(
                model,
                data,
                q=q,
                qd=qd,
                u=u,
                theta=_th,
                coeffs=_co,
                gamma=_g,
                theta_cache=_c,
            )

        ok, wall, dist = cem_plan_oracle(
            predict,
            q0=task["q0"],
            qd0=task["qd0"],
            theta=task["theta"],
            q_star=task["q_star"],
            q_abs_max=task["q_abs_max"],
            u_abs_max=u_abs_max,
            n_cand=n_cand,
            n_iters=n_iters,
            rng=rng,
        )
        oks.append(ok)
        walls.append(wall)
        dists.append(dist if np.isfinite(dist) else float("nan"))
    finite = [d for d in dists if np.isfinite(d)]
    s = float(np.mean(oks)) if oks else 0.0
    return {
        "n_cand": n_cand,
        "n_iters": n_iters,
        "S_oracle": s,
        "mean_dist": float(np.mean(finite)) if finite else float("nan"),
        "cem_wall_mean_s": float(np.mean(walls)) if walls else float("nan"),
        "n_tasks": len(tasks),
        "n_success": int(np.sum(oks)),
    }


def run_cap_x2_p0(
    output: str | Path | None = None,
    *,
    config: CAPX2P0Config | None = None,
) -> dict[str, Any]:
    cfg = config or CAPX2P0Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("CAP-X2-P0 must not write under runs/r10_c0/")
    root.mkdir(parents=True, exist_ok=True)

    print("CAP-X2-P0: building reachable-target tasks (rho=0)...", flush=True)
    tasks = _make_tasks(cfg)
    # Sanity: oracle open-loop u_oracle should reach q_star exactly under same integrator.
    model, data = load_capx_arm3(cfg.physics_timestep)
    t0 = tasks[0]
    cache: dict[str, Any] = {}

    def pred0(q, qd, u, _tv, _th=t0["theta_obj"], _co=t0["coeffs"], _c=cache):
        return oracle_qdd(
            model, data, q=q, qd=qd, u=u, theta=_th, coeffs=_co, gamma=0.0, theta_cache=_c
        )

    q_chk = _rollout_to_target(
        pred0, q0=t0["q0"], qd0=t0["qd0"], theta=t0["theta"], u_seq=t0["u_oracle"]
    )
    reach_err = float(np.max(np.abs(q_chk - t0["q_star"])))

    budget_rows: list[dict[str, Any]] = []
    frozen: dict[str, Any] | None = None
    for n_cand, n_iters in BUDGET_GRID:
        print(f"CAP-X2-P0: try budget cand={n_cand} iters={n_iters}...", flush=True)
        row = evaluate_budget(
            tasks,
            n_cand=n_cand,
            n_iters=n_iters,
            u_abs_max=cfg.u_abs_max,
            physics_timestep=cfg.physics_timestep,
            seed=cfg.task_seed + n_cand * 10 + n_iters,
        )
        budget_rows.append(row)
        _write_json(root / f"budget_{n_cand}_{n_iters}.json", row)
        print(
            f"  S_oracle={row['S_oracle']:.3f} mean_dist={row['mean_dist']:.3f} "
            f"wall={row['cem_wall_mean_s']:.2f}s",
            flush=True,
        )
        if frozen is None and row["S_oracle"] >= ORACLE_SUCCESS_MIN:
            frozen = {
                "cem_candidates": n_cand,
                "cem_iters": n_iters,
                "S_oracle": row["S_oracle"],
                "planning_enabled": True,
            }
            # Do not enlarge further once a passing budget is found.
            break

    if frozen is None:
        best = max(budget_rows, key=lambda r: r["S_oracle"])
        frozen = {
            "cem_candidates": None,
            "cem_iters": None,
            "S_oracle": best["S_oracle"],
            "planning_enabled": False,
            "reason": "max budget S_oracle < 0.80; planning disabled for X2",
            "best_tried": best,
        }

    summary = {
        "stage": "CAP-X2-P0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "residual_family_id": RESIDUAL_FAMILY_ID,
        "rho": float(cfg.rho),
        "unlocks_r10_c0": False,
        "inspects_capacity_curves": False,
        "reachable_target": True,
        "oracle_openloop_reach_err": reach_err,
        "oracle_success_min": ORACLE_SUCCESS_MIN,
        "budget_grid": [{"n_cand": a, "n_iters": b} for a, b in BUDGET_GRID],
        "budget_results": budget_rows,
        "frozen_planner": frozen,
        "planning_component": "enabled" if frozen["planning_enabled"] else "disabled",
        "matched_rule_for_x2": (
            "M1∧M2∧M3" if frozen["planning_enabled"] else "M1∧M2 (predictive only)"
        ),
        "config": asdict(cfg),
        "cap_x2_p0_passed": True,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "FROZEN_PLANNER.json", frozen)
    return summary
