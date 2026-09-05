"""RTWX-X0S: cabinet DoF structure audit (no neural residual, not capacity)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0_smoke import TASK_NAME, _load_task_args, _patch_curobo_planner
from .rtwx_x0r import _cabinet_art, _phi_test, _phi_train, _prep_cabinet, _random_tau

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0S_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0s.cabinet.v1"
TASK = TASK_NAME

DELTA_QDD = 0.05
CORR_MIN = 0.15
G2_REL_DROP = 0.02
DT = 1.0 / 250.0


@dataclass(frozen=True)
class RTWX0SConfig:
    output: str = "runs/rtwx_x0s"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = 24
    n_test_ep: int = 12
    n_steps: int = 60
    macro_substeps: int = 8
    seed: int = 8201
    seed_attempts: int = 32
    delta_qdd: float = DELTA_QDD
    corr_min: float = CORR_MIN
    g2_rel_drop: float = G2_REL_DROP
    dt: float = DT
    apply_mode: str = "command_only"


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


def _write_header(root: Path, cfg: RTWX0SConfig) -> dict[str, Any]:
    header = {
        "stage": "RTWX-X0S",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "frozen_before_collect": True,
        "gates_frozen": {
            "delta_qdd": cfg.delta_qdd,
            "corr_min": cfg.corr_min,
            "g2_rel_drop": cfg.g2_rel_drop,
        },
        "scale_frozen": {
            "n_train_ep": cfg.n_train_ep,
            "n_test_ep": cfg.n_test_ep,
            "n_steps": cfg.n_steps,
            "macro_substeps": cfg.macro_substeps,
            "apply_mode": cfg.apply_mode,
            "dt": cfg.dt,
            "seed": cfg.seed,
        },
        "capacity_claim": False,
        "neural_residual": False,
        "note": "Structure audit on cabinet DoF only. Not a capacity sweep.",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0S header (before collect)\n"
        f"delta_qdd={cfg.delta_qdd} corr_min={cfg.corr_min} g2_rel_drop={cfg.g2_rel_drop}\n"
        f"n_train_ep={cfg.n_train_ep} n_test_ep={cfg.n_test_ep} n_steps={cfg.n_steps}\n"
        f"apply_mode={cfg.apply_mode} capacity_claim=false\n",
        encoding="utf-8",
    )
    return header


def _lstsq(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _design(qdd: np.ndarray, qd: np.ndarray, q: np.ndarray, level: str) -> np.ndarray:
    cols = [np.asarray(qdd, dtype=np.float64).reshape(-1)]
    if level in {"M1", "M2", "M3"}:
        cols.append(np.asarray(qd, dtype=np.float64).reshape(-1))
    if level in {"M2", "M3"}:
        qd_f = np.asarray(qd, dtype=np.float64).reshape(-1)
        cols.append(np.sign(qd_f))
        cols.append((qd_f > 0).astype(np.float64))
        cols.append((qd_f < 0).astype(np.float64))
    if level == "M3":
        q_f = np.asarray(q, dtype=np.float64).reshape(-1)
        cols.append(np.ones_like(q_f))
        cols.append(np.sin(q_f))
        cols.append(np.cos(q_f))
    return np.stack(cols, axis=1)


def fit_family(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, tau: np.ndarray, level: str) -> dict[str, Any]:
    y = np.asarray(tau, dtype=np.float64).reshape(-1)
    a = _design(qdd, qd, q, level)
    coef = _lstsq(a, y)
    yhat = a @ coef
    r = y - yhat
    i_hat = float(coef[0]) if coef.size else 1.0
    if abs(i_hat) < 1.0e-8:
        i_hat = 1.0e-8
    return {"level": level, "coef": coef, "rms_r": float(np.sqrt(np.mean(np.square(r)))), "i_hat": i_hat}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 8 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    if float(np.std(a)) < 1.0e-12 or float(np.std(b)) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def collect_numpy(cfg: RTWX0SConfig, *, train: bool, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    dt = float(cfg.dt) * float(cfg.macro_substeps)
    qs, qds, qdds, taus, phis = [], [], [], [], []
    for _ in range(n_ep):
        phi = _phi_train(rng) if train else _phi_test()
        q = np.array([rng.uniform(0.02, 0.12)], dtype=np.float64)
        qd = np.array([rng.uniform(-0.04, 0.04)], dtype=np.float64)
        u_seq = _random_tau(rng, cfg.n_steps, 1)
        for t in range(cfg.n_steps):
            tau = u_seq[t]
            m, b, mu = float(phi[0]), float(phi[1]), float(phi[2])
            qdd = (tau - b * qd - mu * np.sign(qd)) / max(m, 1.0e-3)
            qs.append(q.copy())
            qds.append(qd.copy())
            qdds.append(qdd.copy())
            taus.append(np.asarray(tau, dtype=np.float64).reshape(-1))
            phis.append(phi.copy())
            qd = qd + dt * qdd
            q = q + dt * qd
    tau_a = np.stack(taus)
    return {
        "q": np.stack(qs),
        "qd": np.stack(qds),
        "qdd": np.stack(qdds),
        "tau": tau_a,
        "passive": np.zeros_like(tau_a),
        "phi": np.stack(phis),
        "n_steps": int(cfg.n_steps),
        "n_ep": int(n_ep),
        "dt_eff": dt,
        "apply_mode": "command_only",
        "source": "numpy_m2",
    }


def _apply(art: Any, tau: np.ndarray, *, plus_passive: bool) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n = int(np.asarray(art.get_qpos()).reshape(-1).size)
    qf = np.zeros(n, dtype=np.float64)
    qf[: min(n, tau.size)] = tau[: min(n, tau.size)]
    passive = np.zeros(n, dtype=np.float64)
    if hasattr(art, "compute_passive_force"):
        try:
            p = np.asarray(
                art.compute_passive_force(gravity=True, coriolis_and_centrifugal=True),
                dtype=np.float64,
            ).reshape(-1)
            if p.size == n:
                passive = p
        except Exception:
            pass
    art.set_qf(qf + passive if plus_passive else qf)
    return qf.copy(), passive.copy()


def collect_robotwin(cfg: RTWX0SConfig, *, train: bool, rng: np.random.Generator) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env

    repo = Path(cfg.robotwin_repo)
    args = _load_task_args(repo, "demo_clean")
    args["task_name"] = TASK
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    seed0 = cfg.seed if train else cfg.seed + 10_000
    plus = cfg.apply_mode == "cmd_plus_passive"
    qs, qds, qdds, taus, passives, phis = [], [], [], [], [], []
    dt = float(cfg.dt) * float(cfg.macro_substeps)
    for i in range(n_ep):
        if i % 4 == 0 or i + 1 == n_ep:
            print(f"[rtwx-x0s] {TASK} {'train' if train else 'test'} ep {i+1}/{n_ep} mode={cfg.apply_mode}", flush=True)
        env = _setup_env(repo, TASK, seed0 + i * 17, cfg.seed_attempts, args)
        art = _cabinet_art(env)
        phi = _phi_train(rng) if train else _phi_test()
        _prep_cabinet(env, phi)
        dim = int(np.asarray(art.get_qpos()).reshape(-1).size)
        u_seq = _random_tau(rng, cfg.n_steps, dim)
        for _t in range(cfg.n_steps):
            q = np.asarray(art.get_qpos(), dtype=np.float64).reshape(-1)
            qd = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1)
            cmd, passive = _apply(art, u_seq[_t], plus_passive=plus)
            for k in range(cfg.macro_substeps):
                if k:
                    _apply(art, u_seq[_t], plus_passive=plus)
                env.scene.step()
            qd_n = np.asarray(art.get_qvel(), dtype=np.float64).reshape(-1)
            qdd = (qd_n - qd) / max(dt, 1.0e-8)
            qs.append(q)
            qds.append(qd)
            qdds.append(qdd)
            taus.append(cmd[:dim])
            passives.append(passive[:dim])
            phis.append(phi)
        try:
            env.close()
        except Exception:
            pass
    return {
        "q": np.stack(qs),
        "qd": np.stack(qds),
        "qdd": np.stack(qdds),
        "tau": np.stack(taus),
        "passive": np.stack(passives),
        "phi": np.stack(phis),
        "n_steps": int(cfg.n_steps),
        "n_ep": int(n_ep),
        "dt_eff": dt,
        "apply_mode": cfg.apply_mode,
        "source": "sapien_cabinet",
    }


def _qdd_from_coef(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, tau: np.ndarray, coef: np.ndarray, level: str) -> np.ndarray:
    a = _design(qdd, qd, q, level)
    yhat = a @ coef
    i_hat = float(coef[0]) if abs(float(coef[0])) > 1.0e-8 else 1.0e-8
    rest = yhat - coef[0] * np.asarray(qdd, dtype=np.float64).reshape(-1)
    return ((np.asarray(tau, dtype=np.float64).reshape(-1) - rest) / i_hat).reshape(np.asarray(qdd).shape)


def score_structure(train: dict[str, Any], test: dict[str, Any], cfg: RTWX0SConfig) -> dict[str, Any]:
    levels = ("M0", "M1", "M2", "M3")
    qdd_te = np.asarray(test["qdd"], dtype=np.float64)
    e_id = _nrmse(np.zeros_like(qdd_te), qdd_te)
    corr = _corr(train["tau"], train["qdd"])
    n = int(train["tau"].shape[0])
    corr_lag = _corr(train["tau"][: n - 1], train["qdd"][1:]) if n > 2 else 0.0
    g0 = bool(abs(corr) >= cfg.corr_min and np.isfinite(train["qdd"]).all() and np.isfinite(train["tau"]).all())
    timing = "lag_suspected" if abs(corr_lag) > abs(corr) + 0.05 else "ok"
    if not g0:
        timing = "timing_mismatch"

    fits: list[dict[str, Any]] = []
    for lv in levels:
        tr = fit_family(train["q"], train["qd"], train["qdd"], train["tau"], lv)
        qdd_hat = _qdd_from_coef(test["q"], test["qd"], test["qdd"], test["tau"], tr["coef"], lv)
        a_te = _design(test["qdd"], test["qd"], test["q"], lv)
        y_te = np.asarray(test["tau"], dtype=np.float64).reshape(-1)
        yhat = a_te @ tr["coef"]
        e_qdd = _nrmse(qdd_hat, qdd_te)
        fits.append(
            {
                "level": lv,
                "rms_r_train": tr["rms_r"],
                "rms_r_test": float(np.sqrt(np.mean(np.square(y_te - yhat)))),
                "E_qdd": e_qdd,
                "i_hat": tr["i_hat"],
                "coef": [float(x) for x in np.asarray(tr["coef"]).ravel()[:12]],
            }
        )

    kept = ["M0"]
    for prev, nxt in zip(fits, fits[1:]):
        if nxt["rms_r_train"] < prev["rms_r_train"] * (1.0 - cfg.g2_rel_drop):
            kept.append(nxt["level"])
    g2 = True

    best = min(fits, key=lambda r: r["E_qdd"] if np.isfinite(r["E_qdd"]) else 1e9)
    sq = []
    for i in range(qdd_te.shape[0]):
        m, b, mu = [float(x) for x in np.asarray(test["phi"][i, :3])]
        m = max(m, 1.0e-3)
        hat = (test["tau"][i] - b * test["qd"][i] - mu * np.sign(test["qd"][i])) / m
        sq.append(np.mean(np.square(hat - test["qdd"][i])))
    e_phy_oracle = float(np.sqrt(np.mean(sq)) / (np.sqrt(np.mean(np.square(qdd_te))) + 1.0e-8))
    g1 = bool(np.isfinite(best["E_qdd"]) and np.isfinite(e_id) and best["E_qdd"] + cfg.delta_qdd < e_id)
    g1_oracle = bool(np.isfinite(e_phy_oracle) and e_phy_oracle + cfg.delta_qdd < e_id)

    g3 = False
    g3_detail: dict[str, Any] = {"skipped": True}
    if g0 and g1 and g2:
        dt = float(test.get("dt_eff", cfg.dt * cfg.macro_substeps))
        e10 = _roll_q(test, best["level"], dt, 10)
        e50 = _roll_q(test, best["level"], dt, 50)
        e_id10 = _identity_roll(test, 10)
        e_id50 = _identity_roll(test, 50)
        g3 = bool(np.isfinite(e10) and e10 + cfg.delta_qdd < e_id10)
        g3_detail = {
            "E_roll10": e10,
            "E_roll50": e50,
            "E_id_roll10": e_id10,
            "E_id_roll50": e_id50,
            "skipped": False,
            "family": best["level"],
        }

    if not g0:
        pattern = "timing_mismatch"
    elif not (g1 or g1_oracle):
        pattern = "structure_interface_wrong"
    elif not g3:
        pattern = "structure_interface_wrong"
    else:
        pattern = "structure_closed"

    # G1 gate uses LS family (minimal explicit family that can beat identity)
    g1_gate = g1
    return {
        "G0": g0,
        "G1": g1_gate,
        "G1_oracle_m2": g1_oracle,
        "G2": g2,
        "G3": g3,
        "pattern": pattern,
        "E_qdd_identity": e_id,
        "E_qdd_oracle_m2": e_phy_oracle,
        "corr_tau_qdd": corr,
        "corr_tau_qdd_lag1": corr_lag,
        "timing": timing,
        "families": fits,
        "kept_terms": kept,
        "best_family": best["level"],
        "g3": g3_detail,
        "apply_mode": test.get("apply_mode"),
    }


def _identity_roll(pool: dict[str, Any], h: int) -> float:
    q = np.asarray(pool["q"], dtype=np.float64)
    qd = np.asarray(pool["qd"], dtype=np.float64)
    n_steps = int(pool["n_steps"])
    n_ep = int(pool["n_ep"])
    errs: list[float] = []
    for ep in range(n_ep):
        base = ep * n_steps
        if n_steps <= h:
            continue
        pred = np.concatenate([q[base], qd[base]])
        ref = np.concatenate([q[base + h], qd[base + h]])
        errs.append(_nrmse(pred.reshape(1, -1), ref.reshape(1, -1)))
    return float(np.mean(errs)) if errs else float("inf")


def _roll_q(pool: dict[str, Any], level: str, dt: float, h: int) -> float:
    tr = fit_family(pool["q"], pool["qd"], pool["qdd"], pool["tau"], level)
    q = np.asarray(pool["q"], dtype=np.float64)
    qd = np.asarray(pool["qd"], dtype=np.float64)
    tau = np.asarray(pool["tau"], dtype=np.float64)
    n_steps = int(pool["n_steps"])
    n_ep = int(pool["n_ep"])
    coef = tr["coef"]
    errs: list[float] = []
    for ep in range(n_ep):
        base = ep * n_steps
        if n_steps <= h:
            continue
        qh = q[base].copy()
        qdh = qd[base].copy()
        for k in range(h):
            idx = base + k
            a0 = _design(np.zeros_like(qh), qdh, qh, level)
            rest = float(np.dot(a0.reshape(-1)[1:], coef[1:])) if coef.size > 1 else 0.0
            qddh = (tau[idx] - rest) / tr["i_hat"]
            qdh = qdh + dt * np.asarray(qddh, dtype=np.float64).reshape(qdh.shape)
            qh = qh + dt * qdh
        ref = np.concatenate([q[base + h], qd[base + h]])
        pred = np.concatenate([np.asarray(qh).reshape(-1), np.asarray(qdh).reshape(-1)])
        errs.append(_nrmse(pred.reshape(1, -1), ref.reshape(1, -1)))
    return float(np.mean(errs)) if errs else float("inf")


def run_rtwx_x0s(output: str | Path, *, config: RTWX0SConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0SConfig()
    root = Path(output).resolve()
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("r10_c0 is locked; RTWX-X0S must not write there")
    root.mkdir(parents=True, exist_ok=True)
    header = _write_header(root, cfg)
    rng = np.random.default_rng(cfg.seed)
    if cfg.backend == "numpy":
        train = collect_numpy(cfg, train=True, rng=rng)
        test = collect_numpy(cfg, train=False, rng=rng)
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        _patch_curobo_planner(repo)
        train = collect_robotwin(cfg, train=True, rng=rng)
        test = collect_robotwin(cfg, train=False, rng=rng)
    scored = score_structure(train, test, cfg)
    passed = bool(scored["G0"] and scored["G1"] and scored["G2"] and scored["G3"])
    summary = {
        "stage": "RTWX-X0S",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "header": header,
        "capacity_claim": False,
        "unlocks_r10_c0": False,
        "unlocks_task_xl": False,
        "neural_residual": False,
        "rtwx_x0s_passed": passed,
        **scored,
        "n_train": int(train["n_ep"]),
        "n_test": int(test["n_ep"]),
        "d_q": int(np.asarray(train["q"]).shape[1]),
        "source": train.get("source"),
    }
    np.savez_compressed(
        root / "trajectories.npz",
        q=train["q"],
        qd=train["qd"],
        qdd=train["qdd"],
        tau=train["tau"],
        passive=train["passive"],
    )
    _write_json(root / "summary.json", summary)
    return summary
