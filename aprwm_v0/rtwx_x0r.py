"""RoboTwin-X0R: cabinet excitation & dynamics-interface instrument (not X0 re-score)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0 import _nrmse, _rollout_nrmse, train_mlp
from .rtwx_x0_smoke import TASK_NAME, _load_task_args, _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0R_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0r.cabinet.v1"
TASK = "put_object_cabinet"

# Frozen BEFORE first collect. Identity must sit outside X0 A/C E1~1e-3 copy.
# Do not copy X0 formal E1 (task B 0.158 / A 0.003) as a post-hoc gate.
EPS_EXC = 0.08
DELTA_ID = 0.05
TAU1 = 0.80
TAU10 = 2.50
H_EXC = 10
DT = 1.0 / 250.0
PHI_HARD = np.array([1.0, 0.08, 0.3], dtype=np.float64)


@dataclass(frozen=True)
class RTWX0RConfig:
    output: str = "runs/rtwx_x0r"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"  # "robotwin" | "numpy"
    n_train_ep: int = 48
    n_test_ep: int = 24
    n_steps: int = 80
    n_cal_steps: int = 24
    macro_substeps: int = 8
    seed: int = 8101
    latent_widths: tuple[int, ...] = (16, 32, 64, 128)
    latent_epochs: int = 40
    seed_attempts: int = 32
    eps_exc: float = EPS_EXC
    delta_id: float = DELTA_ID
    tau1: float = TAU1
    tau10: float = TAU10
    h_exc: int = H_EXC
    dt: float = DT


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or type(obj) is np.bool_:
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not np.isfinite(x):
            return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _write_header(root: Path, cfg: RTWX0RConfig) -> dict[str, Any]:
    header = {
        "stage": "RoboTwin-X0R",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "frozen_before_collect": True,
        "gates_frozen": {
            "eps_exc": cfg.eps_exc,
            "delta_id": cfg.delta_id,
            "tau1": cfg.tau1,
            "tau10": cfg.tau10,
            "h_exc": cfg.h_exc,
        },
        "scale_frozen": {
            "n_train_ep": cfg.n_train_ep,
            "n_test_ep": cfg.n_test_ep,
            "n_steps": cfg.n_steps,
            "n_cal_steps": cfg.n_cal_steps,
            "macro_substeps": cfg.macro_substeps,
            "latent_widths": list(cfg.latent_widths),
            "latent_epochs": cfg.latent_epochs,
            "seed": cfg.seed,
            "dt": cfg.dt,
        },
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "unlocks_task_x1": False,
        "note": (
            "Gates frozen a priori so identity-copy (X0 A/C E1~1e-3) cannot pass G0. "
            "Not a re-score of X0 physics_capacity_shift."
        ),
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "\n".join(
            [
                "RoboTwin-X0R frozen header (before collect)",
                f"task={TASK}",
                f"eps_exc={cfg.eps_exc}",
                f"delta_id={cfg.delta_id}",
                f"tau1={cfg.tau1}",
                f"tau10={cfg.tau10}",
                f"h_exc={cfg.h_exc}",
                f"n_train_ep={cfg.n_train_ep}",
                f"n_test_ep={cfg.n_test_ep}",
                f"n_steps={cfg.n_steps}",
                f"n_cal_steps={cfg.n_cal_steps}",
                f"macro_substeps={cfg.macro_substeps}",
                f"latent_widths={list(cfg.latent_widths)}",
                "capacity_claim=false",
                "unlocks_r10=false",
                "unlocks_task_x1=false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return header


def _phi_train(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(0.8, 1.2), rng.uniform(0.04, 0.12), rng.uniform(0.2, 0.4)], dtype=np.float64)


def _phi_test() -> np.ndarray:
    return np.array([1.5, 0.08, 0.5], dtype=np.float64)


def _random_tau(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    u = np.zeros((n, dim), dtype=np.float64)
    for d in range(dim):
        for _ in range(4):
            amp = float(rng.uniform(6.0, 28.0))
            freq = float(rng.uniform(0.08, 1.2))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            u[:, d] += amp * np.sin(2.0 * np.pi * freq * t / n + phase)
    return u


def dt_eff(cfg: RTWX0RConfig) -> float:
    return float(cfg.dt) * float(cfg.macro_substeps)


def physics_step(q: np.ndarray, qd: np.ndarray, tau: np.ndarray, phi: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    m = max(float(phi[0]), 1.0e-3)
    b = float(phi[1])
    mu = float(phi[2])
    qdd = (np.asarray(tau, dtype=np.float64) - b * qd - mu * np.tanh(qd / 0.05)) / m
    qd_n = qd + dt * qdd
    q_n = q + dt * qd_n
    return q_n, qd_n


def physics_predict(s: np.ndarray, u: np.ndarray, phi: np.ndarray, dt: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = s.shape[0]
    dim = s.shape[1] // 2
    q, qd = s[:, :dim], s[:, dim:]
    tau = u[:, :dim] if u.ndim == 2 else u.reshape(n, -1)[:, :dim]
    qn, qdn = physics_step(q, qd, tau, phi, dt)
    return np.concatenate([qn, qdn], axis=1)


def identity_predict(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.asarray(s, dtype=np.float64).copy()


def excitation_score(s: np.ndarray, n_steps: int, h: int) -> float:
    s = np.asarray(s, dtype=np.float64)
    n_ep = int(s.shape[0] // n_steps)
    scale = np.std(s, axis=0) + 1.0e-8
    ratios: list[float] = []
    for ep in range(n_ep):
        row = s[ep * n_steps : (ep + 1) * n_steps]
        if row.shape[0] <= h:
            continue
        deltas = (row[h:] - row[:-h]) / scale
        ratios.extend(np.linalg.norm(deltas, axis=1).tolist())
    if not ratios:
        return 0.0
    return float(np.median(ratios))


def fit_phi(s: np.ndarray, u: np.ndarray, sp: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares (m, b, mu) from (q, qd, tau) → qdd on D_cal."""
    s = np.asarray(s, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    sp = np.asarray(sp, dtype=np.float64)
    dim = s.shape[1] // 2
    qd = s[:, dim:]
    qdn = sp[:, dim:]
    qdd = (qdn - qd) / max(dt, 1.0e-8)
    tau = u[:, :dim]
    # Stack DoFs: qdd * m + qd * b + tanh(qd)*mu = tau
    y = tau.reshape(-1)
    x_qdd = qdd.reshape(-1)
    x_qd = qd.reshape(-1)
    x_fr = np.tanh(x_qd / 0.05)
    a = np.stack([x_qdd, x_qd, x_fr], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    except np.linalg.LinAlgError:
        return PHI_HARD.copy()
    m = float(np.clip(coef[0], 0.2, 8.0))
    b = float(np.clip(coef[1], 0.0, 2.0))
    mu = float(np.clip(coef[2], 0.0, 2.0))
    if not np.isfinite([m, b, mu]).all() or m < 0.2:
        return PHI_HARD.copy()
    return np.array([m, b, mu], dtype=np.float64)


def per_scene_phi_errors(
    pool: dict[str, np.ndarray],
    *,
    dt: float,
    n_cal_steps: int,
    n_steps: int,
) -> dict[str, Any]:
    s = np.asarray(pool["s"], dtype=np.float64)
    u = np.asarray(pool["u"], dtype=np.float64)
    sp = np.asarray(pool["sp"], dtype=np.float64)
    n_ep = int(pool["n_ep"])
    e_hat: list[float] = []
    e_hard: list[float] = []
    hats: list[list[float]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        cal = slice(ep * n_steps, ep * n_steps + n_cal_steps)
        qry = slice(ep * n_steps + n_cal_steps, (ep + 1) * n_steps)
        if s[qry].shape[0] < 4 or s[cal].shape[0] < 4:
            continue
        phi_hat = fit_phi(s[cal], u[cal], sp[cal], dt)
        hats.append(phi_hat.tolist())
        e_hat.append(_nrmse(physics_predict(s[qry], u[qry], phi_hat, dt), sp[qry]))
        e_hard.append(_nrmse(physics_predict(s[qry], u[qry], PHI_HARD, dt), sp[qry]))
    return {
        "E1_phi_hat_median": float(np.median(e_hat)) if e_hat else float("inf"),
        "E1_phi_hard_median": float(np.median(e_hard)) if e_hard else float("inf"),
        "n_scenes": len(e_hat),
        "phi_hat_examples": hats[:5],
        "E1_hat_list": e_hat,
        "E1_hard_list": e_hard,
    }


def probe_articulation(art: Any) -> dict[str, Any]:
    names = [n for n in ("get_qpos", "get_qvel", "get_qacc", "set_qf", "get_qf", "compute_passive_force") if hasattr(art, n)]
    q = np.asarray(art.get_qpos(), dtype=np.float64).reshape(-1) if hasattr(art, "get_qpos") else np.zeros(0)
    qd = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1) if hasattr(art, "get_qvel") else np.zeros(0)
    qdd_api = False
    qdd = None
    if hasattr(art, "get_qacc"):
        try:
            qdd = np.asarray(art.get_qacc(), dtype=np.float64).reshape(-1)
            qdd_api = bool(qdd.size) and np.isfinite(qdd).all()
        except Exception:
            qdd_api = False
    return {
        "methods": names,
        "n_dof": int(q.size),
        "has_q": bool(q.size),
        "has_qdot": bool(qd.size),
        "has_qddot_api": qdd_api,
        "has_set_qf": hasattr(art, "set_qf"),
        "has_passive": hasattr(art, "compute_passive_force"),
    }


def _cabinet_art(env: Any) -> Any:
    cab = env.cabinet
    return getattr(cab, "actor", cab)


def _prep_cabinet(env: Any, phi: np.ndarray) -> None:
    cab = env.cabinet
    art = _cabinet_art(env)
    try:
        cab.set_mass(float(phi[0]))
    except Exception:
        pass
    try:
        cab.set_properties(damping=0.0, stiffness=0.0, friction=float(phi[2]))
    except Exception:
        pass
    if hasattr(art, "set_sleep_threshold"):
        try:
            art.set_sleep_threshold(0.0)
        except Exception:
            pass
    if hasattr(art, "get_active_joints"):
        for joint in art.get_active_joints():
            if hasattr(joint, "set_friction"):
                try:
                    joint.set_friction(float(phi[2]))
                except Exception:
                    pass
            if hasattr(joint, "set_drive_properties"):
                try:
                    joint.set_drive_properties(stiffness=0.0, damping=0.0)
                except Exception:
                    pass


def _apply_tau(art: Any, tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = int(np.asarray(art.get_qpos()).reshape(-1).size)
    qf = np.zeros(n, dtype=np.float64)
    qf[: min(n, tau.size)] = tau[: min(n, tau.size)]
    applied = qf.copy()
    if hasattr(art, "compute_passive_force"):
        try:
            passive = np.asarray(art.compute_passive_force(gravity=True, coriolis_and_centrifugal=True), dtype=np.float64).reshape(-1)
            if passive.size == n:
                art.set_qf(passive + qf)
                return applied
        except Exception:
            pass
    art.set_qf(qf)
    return applied


def _state_from_art(art: Any) -> np.ndarray:
    q = np.asarray(art.get_qpos(), dtype=np.float64).reshape(-1)
    qd = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1)
    return np.concatenate([q, qd])


def collect_numpy(cfg: RTWX0RConfig, *, train: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    dim = 1
    dt = dt_eff(cfg)
    s_rows, sp_rows, u_rows, phi_rows = [], [], [], []
    for _ in range(n_ep):
        phi = _phi_train(rng) if train else _phi_test()
        q = np.array([rng.uniform(0.02, 0.10)], dtype=np.float64)
        qd = np.array([rng.uniform(-0.05, 0.05)], dtype=np.float64)
        u_seq = _random_tau(rng, cfg.n_steps, dim)
        ss, us = [], []
        for t in range(cfg.n_steps):
            ss.append(np.concatenate([q, qd]))
            us.append(u_seq[t])
            q, qd = physics_step(q, qd, u_seq[t], phi, dt)
        ss.append(np.concatenate([q, qd]))
        s = np.stack(ss[:-1], axis=0)
        sp = np.stack(ss[1:], axis=0)
        s_rows.append(s)
        sp_rows.append(sp)
        u_rows.append(u_seq)
        phi_rows.append(np.repeat(phi.reshape(1, -1), cfg.n_steps, axis=0))
    qdd = (np.concatenate(sp_rows)[:, dim:] - np.concatenate(s_rows)[:, dim:]) / dt
    return {
        "s": np.concatenate(s_rows, axis=0),
        "sp": np.concatenate(sp_rows, axis=0),
        "u": np.concatenate(u_rows, axis=0),
        "phi": np.concatenate(phi_rows, axis=0),
        "qdd": qdd,
        "n_steps": int(cfg.n_steps),
        "n_ep": int(n_ep),
        "audit": {
            "has_q": True,
            "has_qdot": True,
            "has_tau": True,
            "has_qddot_api": False,
            "has_qddot_fd": True,
            "n_dof": dim,
            "source": "numpy_plant",
        },
    }


def collect_robotwin(cfg: RTWX0RConfig, *, train: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    repo = Path(cfg.robotwin_repo)
    args = _load_task_args(repo, "demo_clean")
    args["task_name"] = TASK
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    seed0 = cfg.seed if train else cfg.seed + 10_000
    s_rows, sp_rows, u_rows, phi_rows, qdd_rows = [], [], [], [], []
    audit: dict[str, Any] | None = None
    dim_u = 1
    from .rtwx_x0 import _setup_env

    for i in range(n_ep):
        if i % 4 == 0 or i + 1 == n_ep:
            print(f"[rtwx-x0r] {TASK} {'train' if train else 'test'} ep {i+1}/{n_ep}", flush=True)
        env = _setup_env(repo, TASK, seed0 + i * 17, cfg.seed_attempts, args)
        art = _cabinet_art(env)
        if audit is None:
            audit = probe_articulation(art)
            audit["has_tau"] = bool(audit.get("has_set_qf"))
            audit["has_qddot_fd"] = bool(audit.get("has_qdot"))
            audit["source"] = "sapien_cabinet"
        dim = int(np.asarray(art.get_qpos()).reshape(-1).size) if hasattr(art, "get_qpos") else 1
        dim_u = max(dim, 1)
        phi = _phi_train(rng) if train else _phi_test()
        _prep_cabinet(env, phi)
        u_seq = _random_tau(rng, cfg.n_steps, dim_u)
        ss, us, qdd_fd = [], [], []
        qd_prev = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1)
        dt = float(cfg.dt)
        for t in range(cfg.n_steps):
            ss.append(_state_from_art(art))
            applied = np.zeros(dim_u, dtype=np.float64)
            for _ in range(cfg.macro_substeps):
                applied = _apply_tau(art, u_seq[t])
                env.scene.step()
            us.append(applied.reshape(-1)[:dim_u])
            qd_now = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1)
            qdd_fd.append((qd_now - qd_prev) / max(dt * cfg.macro_substeps, 1e-8))
            qd_prev = qd_now
        ss.append(_state_from_art(art))
        s = np.stack(ss[:-1], axis=0)
        sp = np.stack(ss[1:], axis=0)
        s_rows.append(s)
        sp_rows.append(sp)
        u_rows.append(np.stack(us, axis=0))
        qdd_rows.append(np.stack(qdd_fd, axis=0))
        phi_rows.append(np.repeat(phi.reshape(1, -1), cfg.n_steps, axis=0))
        try:
            env.close()
        except Exception:
            pass
    assert audit is not None
    return {
        "s": np.concatenate(s_rows, axis=0),
        "sp": np.concatenate(sp_rows, axis=0),
        "u": np.concatenate(u_rows, axis=0),
        "phi": np.concatenate(phi_rows, axis=0),
        "qdd": np.concatenate(qdd_rows, axis=0),
        "n_steps": int(cfg.n_steps),
        "n_ep": int(n_ep),
        "audit": audit,
    }


def _oracle_e1(pool: dict[str, np.ndarray], dt: float) -> float:
    s, u, sp, phi = pool["s"], pool["u"], pool["sp"], pool["phi"]
    n_steps = int(pool["n_steps"])
    n_ep = int(pool["n_ep"])
    errs = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        p = phi[sl][0]
        errs.append(_nrmse(physics_predict(s[sl], u[sl], p, dt), sp[sl]))
    return float(np.median(errs)) if errs else float("inf")


def _latent_delta_train(pool: dict[str, np.ndarray], *, hidden: int, epochs: int, seed: int):
    def residual_of(s, u):
        return np.asarray(s, dtype=np.float64)

    pred, n_param = train_mlp(
        {"s": pool["s"], "u": pool["u"], "sp": pool["sp"]},
        hidden=hidden,
        out_dim=int(pool["s"].shape[1]),
        epochs=epochs,
        seed=seed,
        residual_of=residual_of,
    )
    return pred, n_param


def score_gates(
    *,
    cfg: RTWX0RConfig,
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    latent_curve: list[dict[str, Any]],
    g1_audit: dict[str, Any],
) -> dict[str, Any]:
    dt = dt_eff(cfg)
    exc = excitation_score(test["s"], int(test["n_steps"]), cfg.h_exc)
    e_id = _nrmse(identity_predict(test["s"], test["u"]), test["sp"])
    e_phy = _oracle_e1(test, dt)
    g0 = bool(exc > cfg.eps_exc and np.isfinite(e_id) and np.isfinite(e_phy) and e_id > e_phy + cfg.delta_id)
    g1 = bool(
        g1_audit.get("has_q")
        and g1_audit.get("has_qdot")
        and g1_audit.get("has_tau")
        and (g1_audit.get("has_qddot_api") or g1_audit.get("has_qddot_fd"))
        and int(g1_audit.get("n_dof") or 0) >= 1
    )
    phi_stats = per_scene_phi_errors(test, dt=dt, n_cal_steps=cfg.n_cal_steps, n_steps=int(test["n_steps"]))
    g2 = bool(
        g1
        and np.isfinite(phi_stats["E1_phi_hat_median"])
        and np.isfinite(phi_stats["E1_phi_hard_median"])
        and phi_stats["E1_phi_hat_median"] < phi_stats["E1_phi_hard_median"]
    )
    g3 = False
    g3_detail: dict[str, Any] = {"reason": "no_latent"}
    if latent_curve:
        largest = max(latent_curve, key=lambda r: int(r["hidden"]))
        e1s = [float(r["E1"]) for r in latent_curve]
        rolls = [float(r["E_roll10"]) for r in latent_curve]
        best_e1_i = int(np.argmin(e1s))
        worst_roll_i = int(np.argmax(rolls))
        same_extreme = best_e1_i == worst_roll_i and len(latent_curve) > 1
        finite_ok = bool(np.isfinite(largest["E1"]) and np.isfinite(largest["E_roll10"]))
        bound_ok = bool(float(largest["E1"]) < cfg.tau1 and float(largest["E_roll10"]) < cfg.tau10)
        g3 = bool(finite_ok and bound_ok and not same_extreme)
        g3_detail = {
            "largest_hidden": int(largest["hidden"]),
            "largest_E1": float(largest["E1"]),
            "largest_E_roll10": float(largest["E_roll10"]),
            "best_e1_hidden": int(latent_curve[best_e1_i]["hidden"]),
            "worst_roll_hidden": int(latent_curve[worst_roll_i]["hidden"]),
            "best_e1_is_worst_rollout": same_extreme,
            "finite": finite_ok,
            "within_tau": bound_ok,
        }
    if not g1:
        pattern = "not_force_auditable"
    elif not g0:
        pattern = "excitation_failure"
    elif not g2:
        pattern = "phi_unidentified"
    elif not g3:
        pattern = "latent_rollout_unstable"
    else:
        pattern = "instrument_ready"
    return {
        "G0": g0,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "pattern": pattern,
        "metrics": {
            "excitation_median": exc,
            "E_identity": e_id,
            "E_oracle_physics": e_phy,
            "phi": phi_stats,
            "g3": g3_detail,
        },
    }


def run_rtwx_x0r(output: str | Path, *, config: RTWX0RConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0RConfig()
    root = Path(output).resolve()
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("r10_c0 is locked; RoboTwin-X0R must not write there")
    root.mkdir(parents=True, exist_ok=True)
    header = _write_header(root, cfg)

    summary: dict[str, Any] = {
        "stage": "RoboTwin-X0R",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "rgb_in_s": False,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "unlocks_task_x1": False,
        "official_robotwin_task_executed": False,
        "header": header,
        "rtwx_x0r_passed": False,
        "instrument_ready": False,
        "pattern": None,
    }

    rng = np.random.default_rng(cfg.seed)
    if cfg.backend == "numpy":
        train = collect_numpy(cfg, train=True, rng=rng)
        test = collect_numpy(cfg, train=False, rng=np.random.default_rng(cfg.seed + 1))
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        _patch_curobo_planner(repo)
        train = collect_robotwin(cfg, train=True, rng=rng)
        test = collect_robotwin(cfg, train=False, rng=np.random.default_rng(cfg.seed + 1))
        summary["official_robotwin_task_executed"] = True

    np.savez_compressed(
        root / "trajectories.npz",
        train_s=train["s"],
        train_u=train["u"],
        train_sp=train["sp"],
        test_s=test["s"],
        test_u=test["u"],
        test_sp=test["sp"],
        test_phi=test["phi"],
        test_qdd=test["qdd"],
    )

    dt = dt_eff(cfg)
    out_dim = int(train["s"].shape[1])
    latent_curve = []
    for h in cfg.latent_widths:
        pred, n_param = _latent_delta_train(train, hidden=h, epochs=cfg.latent_epochs, seed=cfg.seed + h)
        e1 = _nrmse(pred(test["s"], test["u"]), test["sp"])
        roll = _rollout_nrmse(pred, test, 10)
        latent_curve.append({"hidden": h, "P": n_param, "E1": e1, "E_roll10": roll})

    scored = score_gates(cfg=cfg, train=train, test=test, latent_curve=latent_curve, g1_audit=train["audit"])
    pattern = scored["pattern"]
    passed = pattern == "instrument_ready"
    summary.update(
        {
            "d_s": out_dim,
            "d_u": int(train["u"].shape[1]),
            "backend": cfg.backend,
            "audit": train["audit"],
            "gates": {k: scored[k] for k in ("G0", "G1", "G2", "G3")},
            "metrics": scored["metrics"],
            "latent": latent_curve,
            "dt_eff": dt,
            "pattern": pattern,
            "rtwx_x0r_passed": passed,
            "instrument_ready": passed,
            "output": str(root),
        }
    )
    _write_json(root / "summary.json", summary)
    _write_json(root / "latent.json", {"latent": latent_curve})
    return summary
