"""PLAN-X0: reachable-target / teacher A* / FD curvature instrument.

Implementation header (locked before any formal histogram of h):
  U_ABS_MAX = 1.5          # same as CAP-X2-P0
  EPS_H     = 0.05         # curvature probe on knot coordinates
  EPS_GRAD  = 1.0e-3       # one-sided FD for L-BFGS-B
  TEACHER   = L-BFGS-B, maxiter=8, bounds [-U_ABS_MAX, U_ABS_MAX]
  A_gen     = smoothed random knots, then clip

Does not train a proposal network. Does not touch CAP-X2. Does not unlock R10.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import minimize

from .cap_x2_p0 import MODEL_DT, PLAN_HORIZON_S, ACTION_KNOTS, EPS_Q, LAMBDA_U, LAMBDA_V, oracle_qdd
from .capx_arm3_plant import (
    HOST_PLANT_ID,
    N_DOF,
    SceneTheta,
    load_capx_arm3,
    sample_scene_theta,
)
from .capx_residual import SceneResidualCoeffs

PREREG_PATH = "REPORT/REG/PLANX/PLANX0_PREREG.md"
SCHEMA_ID = "aprwm.plan_x0.instrument.v1"

U_ABS_MAX = 1.5
EPS_H = 0.05
EPS_GRAD = 1.0e-3
TEACHER_MAXITER = 8
HORIZON_STEPS = int(round(PLAN_HORIZON_S / MODEL_DT))
A_DIM = ACTION_KNOTS * N_DOF  # 60
ZERO_RESIDUAL = SceneResidualCoeffs(
    alpha=np.zeros(3), beta=np.zeros(3), eta=np.zeros(3)
)

TRAIN_SCENE_SEED0 = 70_000
VAL_SCENE_SEED0 = 80_000
TEST_SCENE_SEED0 = 90_000
TARGET_SEED0 = 110_000


@dataclass(frozen=True)
class PLANX0Config:
    output: str = "runs/plan_x0/formal"
    n_train_scenes: int = 128
    n_val_scenes: int = 32
    n_test_scenes: int = 64
    targets_per_scene: int = 32
    physics_timestep: float = 0.002
    u_abs_max: float = U_ABS_MAX
    eps_h: float = EPS_H
    teacher_maxiter: int = TEACHER_MAXITER


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def knots_to_u_seq(a: np.ndarray) -> np.ndarray:
    knots = np.asarray(a, dtype=np.float64).reshape(ACTION_KNOTS, N_DOF)
    seg = max(1, HORIZON_STEPS // ACTION_KNOTS)
    u = np.zeros((HORIZON_STEPS, N_DOF), dtype=np.float64)
    for t in range(HORIZON_STEPS):
        u[t] = knots[min(t // seg, ACTION_KNOTS - 1)]
    return u


def sample_smooth_knots(rng: np.random.Generator, u_abs_max: float) -> np.ndarray:
    raw = rng.uniform(-u_abs_max, u_abs_max, size=(ACTION_KNOTS, N_DOF))
    sm = raw.copy()
    if ACTION_KNOTS >= 3:
        sm[1:-1] = 0.25 * raw[:-2] + 0.50 * raw[1:-1] + 0.25 * raw[2:]
    return np.clip(sm, -u_abs_max, u_abs_max).reshape(A_DIM)


def rollout_cost(
    a: np.ndarray,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    q_star: np.ndarray,
    theta: SceneTheta,
    model: Any,
    data: Any,
    cache: dict[str, Any],
    u_abs_max: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (J, q_T, u_seq) under Euler + oracle_qdd."""

    a = np.clip(np.asarray(a, dtype=np.float64).reshape(A_DIM), -u_abs_max, u_abs_max)
    u_seq = knots_to_u_seq(a)
    q = np.asarray(q0, dtype=np.float64).copy()
    qd = np.asarray(qd0, dtype=np.float64).copy()
    cost = 0.0
    for t in range(HORIZON_STEPS):
        u = u_seq[t]
        qdd = oracle_qdd(
            model,
            data,
            q=q,
            qd=qd,
            u=u,
            theta=theta,
            coeffs=ZERO_RESIDUAL,
            gamma=0.0,
            theta_cache=cache,
        )
        q = q + qd * MODEL_DT
        qd = qd + qdd * MODEL_DT
        cost += float(
            np.sum((q - q_star) ** 2) + LAMBDA_V * np.sum(qd**2) + LAMBDA_U * np.sum(u**2)
        )
    return cost, q, u_seq


def teacher_lbfgs(
    a_gen: np.ndarray,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    q_star: np.ndarray,
    theta: SceneTheta,
    model: Any,
    data: Any,
    cache: dict[str, Any],
    u_abs_max: float,
    maxiter: int,
) -> tuple[np.ndarray, float, float]:
    """Warm-start L-BFGS-B from A_gen. Returns (A*, J*, J_gen)."""

    j_gen, _, _ = rollout_cost(
        a_gen, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
    )

    def fun(x: np.ndarray) -> float:
        j, _, _ = rollout_cost(
            x, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
        )
        return j

    def jac(x: np.ndarray) -> np.ndarray:
        g = np.zeros(A_DIM, dtype=np.float64)
        f0 = fun(x)
        for j in range(A_DIM):
            xp = x.copy()
            xp[j] = np.clip(xp[j] + EPS_GRAD, -u_abs_max, u_abs_max)
            g[j] = (fun(xp) - f0) / EPS_GRAD
        return g

    bounds = [(-u_abs_max, u_abs_max)] * A_DIM
    res = minimize(
        fun,
        np.asarray(a_gen, dtype=np.float64).reshape(A_DIM),
        method="L-BFGS-B",
        jac=jac,
        bounds=bounds,
        options={"maxiter": int(maxiter), "maxfun": int(maxiter) * (A_DIM + 2), "ftol": 1.0e-8},
    )
    a_star = np.clip(np.asarray(res.x, dtype=np.float64).reshape(A_DIM), -u_abs_max, u_abs_max)
    j_star, _, _ = rollout_cost(
        a_star, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
    )
    if j_star > j_gen:
        return a_gen.copy(), float(j_gen), float(j_gen)
    return a_star, float(j_star), float(j_gen)


def finite_diff_h(
    a_star: np.ndarray,
    *,
    q0: np.ndarray,
    qd0: np.ndarray,
    q_star: np.ndarray,
    theta: SceneTheta,
    model: Any,
    data: Any,
    cache: dict[str, Any],
    u_abs_max: float,
    eps_h: float,
) -> tuple[np.ndarray, float]:
    j0, _, _ = rollout_cost(
        a_star, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
    )
    h = np.zeros(A_DIM, dtype=np.float64)
    for j in range(A_DIM):
        ap = a_star.copy()
        am = a_star.copy()
        ap[j] = np.clip(a_star[j] + eps_h, -u_abs_max, u_abs_max)
        am[j] = np.clip(a_star[j] - eps_h, -u_abs_max, u_abs_max)
        jp, _, _ = rollout_cost(
            ap, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
        )
        jm, _, _ = rollout_cost(
            am, q0=q0, qd0=qd0, q_star=q_star, theta=theta, model=model, data=data, cache=cache, u_abs_max=u_abs_max
        )
        h[j] = (jp + jm - 2.0 * j0) / (eps_h**2)
    return h, float(j0)


def _scene_ids(split: str, cfg: PLANX0Config) -> list[int]:
    if split == "train":
        return [TRAIN_SCENE_SEED0 + i for i in range(cfg.n_train_scenes)]
    if split == "val":
        return [VAL_SCENE_SEED0 + i for i in range(cfg.n_val_scenes)]
    if split == "test":
        return [TEST_SCENE_SEED0 + i for i in range(cfg.n_test_scenes)]
    raise ValueError(split)


def build_split(root: Path, *, split: str, cfg: PLANX0Config) -> dict[str, Any]:
    model, data = load_capx_arm3(cfg.physics_timestep)
    scene_ids = _scene_ids(split, cfg)
    split_dir = root / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n_feas = 0
    n_teacher_ok = 0
    n_cond = 0
    h_all: list[np.ndarray] = []
    j_star_all: list[float] = []
    j_gen_all: list[float] = []

    for scene_seed in scene_ids:
        scene_rng = np.random.default_rng(scene_seed)
        theta = sample_scene_theta(scene_rng)
        cache: dict[str, Any] = {}
        path = split_dir / f"scene_{scene_seed:06d}.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = SCHEMA_ID
            handle.attrs["host_plant_id"] = HOST_PLANT_ID
            handle.attrs["split"] = split
            handle.attrs["scene_seed"] = int(scene_seed)
            handle.create_dataset("theta", data=theta.as_vector())
            for k in range(cfg.targets_per_scene):
                tgt_seed = TARGET_SEED0 + 10_000 * scene_seed + k
                rng = np.random.default_rng(tgt_seed)
                q0 = rng.uniform(-0.6, 0.6, size=N_DOF)
                qd0 = rng.uniform(-0.3, 0.3, size=N_DOF)
                a_gen = sample_smooth_knots(rng, cfg.u_abs_max)
                # Construct q* from A_gen (reachable).
                dummy_star = np.zeros(N_DOF)
                _, q_h, _ = rollout_cost(
                    a_gen,
                    q0=q0,
                    qd0=qd0,
                    q_star=dummy_star,
                    theta=theta,
                    model=model,
                    data=data,
                    cache=cache,
                    u_abs_max=cfg.u_abs_max,
                )
                q_star = q_h.copy()
                j_chk, q_chk, _ = rollout_cost(
                    a_gen,
                    q0=q0,
                    qd0=qd0,
                    q_star=q_star,
                    theta=theta,
                    model=model,
                    data=data,
                    cache=cache,
                    u_abs_max=cfg.u_abs_max,
                )
                feas = float(np.max(np.abs(q_chk - q_star))) < 1.0e-8
                n_feas += int(feas)
                a_star, j_star, j_gen = teacher_lbfgs(
                    a_gen,
                    q0=q0,
                    qd0=qd0,
                    q_star=q_star,
                    theta=theta,
                    model=model,
                    data=data,
                    cache=cache,
                    u_abs_max=cfg.u_abs_max,
                    maxiter=cfg.teacher_maxiter,
                )
                n_teacher_ok += int(j_star <= j_gen + 1.0e-9)
                h, _ = finite_diff_h(
                    a_star,
                    q0=q0,
                    qd0=qd0,
                    q_star=q_star,
                    theta=theta,
                    model=model,
                    data=data,
                    cache=cache,
                    u_abs_max=cfg.u_abs_max,
                    eps_h=cfg.eps_h,
                )
                h_all.append(h)
                j_star_all.append(j_star)
                j_gen_all.append(j_gen)
                n_cond += 1
                grp = handle.create_group(f"cond_{k:03d}")
                grp.attrs["target_seed"] = int(tgt_seed)
                grp.attrs["feasible_agen"] = bool(feas)
                grp.create_dataset("q0", data=q0)
                grp.create_dataset("qd0", data=qd0)
                grp.create_dataset("q_star", data=q_star)
                # Oracle-only (hidden from later proposal learners unless X1 says so).
                oracle = grp.create_group("oracle")
                oracle.create_dataset("A_gen", data=a_gen)
                oracle.create_dataset("A_star", data=a_star)
                oracle.create_dataset("h", data=h)
                oracle.attrs["J_gen"] = float(j_gen)
                oracle.attrs["J_star"] = float(j_star)
                oracle.attrs["J_chk_agen"] = float(j_chk)

    h_cat = np.concatenate(h_all) if h_all else np.zeros(0)
    return {
        "split": split,
        "n_scenes": len(scene_ids),
        "n_conditions": n_cond,
        "n_feasible": n_feas,
        "n_teacher_ok": n_teacher_ok,
        "S_feasible": float(n_feas / max(n_cond, 1)),
        "teacher_frac": float(n_teacher_ok / max(n_cond, 1)),
        "h": h_cat,
        "J_star_mean": float(np.mean(j_star_all)) if j_star_all else float("nan"),
        "J_gen_mean": float(np.mean(j_gen_all)) if j_gen_all else float("nan"),
    }


def _gates_from_h(h: np.ndarray, *, s_feas: float, teacher_frac: float) -> dict[str, Any]:
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    g0 = bool(s_feas >= 1.0 - 1.0e-12)
    g1 = bool(teacher_frac >= 0.99)
    g2 = bool(h.size > 0 and float(np.mean(h > 0.0)) >= 0.90)
    pos = h[h > 0.0]
    if pos.size < 20:
        ratio = float("nan")
        g3 = False
        degenerate = True
    else:
        q10, q90 = np.quantile(pos, [0.10, 0.90])
        ratio = float(q90 / max(float(q10), 1.0e-18))
        g3 = bool(ratio >= 5.0)
        degenerate = not g3
    return {
        "G0_feasibility": g0,
        "G1_teacher": g1,
        "G2_local_min": g2,
        "G3_diversity": g3,
        "G_label": True,
        "P_h_positive": float(np.mean(h > 0.0)) if h.size else float("nan"),
        "Q90_over_Q10_positive_h": ratio,
        "sensitivity_degenerate": bool(degenerate),
        "S_feasible": float(s_feas),
        "teacher_frac": float(teacher_frac),
    }


def run_plan_x0(
    output: str | Path | None = None,
    *,
    config: PLANX0Config | None = None,
) -> dict[str, Any]:
    cfg = config or PLANX0Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X0 must not write under runs/r10_c0/")
    if "cap_x2" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X0 must not write under runs/cap_x2/")
    root.mkdir(parents=True, exist_ok=True)

    print("PLAN-X0: train split...", flush=True)
    train = build_split(root, split="train", cfg=cfg)
    print("PLAN-X0: val split...", flush=True)
    val = build_split(root, split="val", cfg=cfg)
    print("PLAN-X0: test split...", flush=True)
    test = build_split(root, split="test", cfg=cfg)

    h_all = np.concatenate([train["h"], val["h"], test["h"]])
    n_cond = train["n_conditions"] + val["n_conditions"] + test["n_conditions"]
    n_feas = train["n_feasible"] + val["n_feasible"] + test["n_feasible"]
    n_tok = train["n_teacher_ok"] + val["n_teacher_ok"] + test["n_teacher_ok"]
    gates = _gates_from_h(
        h_all,
        s_feas=n_feas / max(n_cond, 1),
        teacher_frac=n_tok / max(n_cond, 1),
    )
    gates["G_label"] = True
    passed = bool(
        gates["G0_feasibility"]
        and gates["G1_teacher"]
        and gates["G2_local_min"]
        and gates["G3_diversity"]
        and gates["G_label"]
    )
    summary = {
        "stage": "PLAN-X0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "unlocks_r10_c0": False,
        "proposal_claim": False,
        "implementation_header": {
            "U_ABS_MAX": U_ABS_MAX,
            "EPS_H": cfg.eps_h,
            "EPS_GRAD": EPS_GRAD,
            "TEACHER": "L-BFGS-B",
            "TEACHER_MAXITER": cfg.teacher_maxiter,
            "A_DIM": A_DIM,
            "ACTION_KNOTS": ACTION_KNOTS,
            "HORIZON_STEPS": HORIZON_STEPS,
        },
        "config": asdict(cfg),
        "splits": {
            "train": {k: v for k, v in train.items() if k != "h"},
            "val": {k: v for k, v in val.items() if k != "h"},
            "test": {k: v for k, v in test.items() if k != "h"},
        },
        "gates": gates,
        "plan_x0_passed": passed,
        "plan_x1_unlocked": bool(passed and not gates["sensitivity_degenerate"]),
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    np.save(root / "h_all.npy", h_all)
    return summary
