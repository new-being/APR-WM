"""RTWX-X0F: force-channel / clock audit (no train, no phi, no capacity)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0_smoke import TASK_NAME, _load_task_args, _patch_curobo_planner
from .rtwx_x0r import _cabinet_art, _phi_test, _phi_train, _prep_cabinet, _random_tau

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0F_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0f.force_clock.v1"
TASK = TASK_NAME
RHO_MIN = 0.30
LAGS = (-2, -1, 0, 1, 2)
CHANNELS = ("tau_cmd", "qf_post_set", "cmd_plus_passive", "qf_post", "cmd_plus_passive_ext")
QDD_SRCS = ("qacc_pre", "qacc_post", "qdd_fd")


@dataclass(frozen=True)
class RTWX0FConfig:
    output: str = "runs/rtwx_x0f"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = 12
    n_test_ep: int = 6
    n_steps: int = 80
    seed: int = 8301
    seed_attempts: int = 32
    rho_min: float = RHO_MIN
    dt: float = 1.0 / 250.0


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


def _write_header(root: Path, cfg: RTWX0FConfig) -> dict[str, Any]:
    header = {
        "stage": "RTWX-X0F",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_before_collect": True,
        "rho_min": cfg.rho_min,
        "lags": list(LAGS),
        "channels": list(CHANNELS),
        "qdd_sources": list(QDD_SRCS),
        "n_train_ep": cfg.n_train_ep,
        "n_test_ep": cfg.n_test_ep,
        "n_steps": cfg.n_steps,
        "one_scene_step_per_log": True,
        "capacity_claim": False,
        "phi_fit": False,
        "neural": False,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        f"RTWX-X0F header (before collect)\nrho_min={cfg.rho_min} lags={list(LAGS)}\n"
        f"n_train_ep={cfg.n_train_ep} n_test_ep={cfg.n_test_ep} n_steps={cfg.n_steps}\n"
        "one_scene_step_per_log=true capacity_claim=false phi_fit=false\n",
        encoding="utf-8",
    )
    return header


def _vec(art: Any, name: str, n: int) -> np.ndarray:
    fn = getattr(art, name, None)
    if fn is None:
        return np.full(n, np.nan)
    try:
        x = np.asarray(fn(), dtype=np.float64).reshape(-1)
        out = np.full(n, np.nan)
        m = min(n, x.size)
        out[:m] = x[:m]
        return out
    except Exception:
        return np.full(n, np.nan)


def _passive(art: Any, n: int) -> np.ndarray:
    if not hasattr(art, "compute_passive_force"):
        return np.zeros(n)
    try:
        p = np.asarray(art.compute_passive_force(gravity=True, coriolis_and_centrifugal=True), dtype=np.float64).reshape(-1)
        out = np.zeros(n)
        out[: min(n, p.size)] = p[: min(n, p.size)]
        return out
    except Exception:
        return np.zeros(n)


def _ext(art: Any, n: int) -> tuple[np.ndarray, bool]:
    for name in (
        "compute_generalized_external_force",
        "compute_external_force",
        "get_external_force",
    ):
        fn = getattr(art, name, None)
        if fn is None:
            continue
        try:
            x = np.asarray(fn(), dtype=np.float64).reshape(-1)
            out = np.zeros(n)
            out[: min(n, x.size)] = x[: min(n, x.size)]
            return out, True
        except Exception:
            continue
    return np.zeros(n), False


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 16 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _min_dof_corr(tau: np.ndarray, qdd: np.ndarray, lag: int, n_steps: int, n_ep: int) -> float:
    dof = tau.shape[1]
    mins: list[float] = []
    for d in range(dof):
        cs: list[float] = []
        for ep in range(n_ep):
            x = tau[ep * n_steps : (ep + 1) * n_steps, d]
            y = qdd[ep * n_steps : (ep + 1) * n_steps, d]
            if lag >= 0:
                x, y = x[: n_steps - lag], y[lag:]
            else:
                x, y = x[-lag:], y[: n_steps + lag]
            cs.append(_corr(x, y))
        mins.append(abs(float(np.mean(cs))))
    return float(min(mins)) if mins else 0.0


def collect_numpy(cfg: RTWX0FConfig, *, train: bool, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    n, dim, dt = cfg.n_steps, 1, cfg.dt
    rows = {k: [] for k in ("tau_cmd", "qf_post_set", "passive", "qf_post", "ext", "qacc_pre", "qacc_post", "qdd_fd")}
    for _ in range(n_ep):
        phi = _phi_train(rng) if train else _phi_test()
        q = np.array([0.05], dtype=np.float64)
        qd = np.zeros(1)
        u = _random_tau(rng, n, dim)
        for t in range(n):
            tau = u[t].reshape(-1)
            qacc_pre = ((tau - float(phi[1]) * qd) / max(float(phi[0]), 1e-3)).reshape(-1)
            rows["tau_cmd"].append(tau)
            rows["qf_post_set"].append(tau)
            rows["passive"].append(float(phi[1]) * qd)
            rows["qf_post"].append(tau)
            rows["ext"].append(np.zeros(dim))
            rows["qacc_pre"].append(qacc_pre)
            qdd = (tau - float(phi[1]) * qd - float(phi[2]) * np.sign(qd)) / max(float(phi[0]), 1e-3)
            qd = qd + dt * qdd
            q = q + dt * qd
            rows["qacc_post"].append(qdd.reshape(-1))
            rows["qdd_fd"].append(qdd.reshape(-1))
    out = {k: np.stack(v) for k, v in rows.items()}
    out.update({"n_steps": n, "n_ep": n_ep, "ext_api": False, "source": "numpy"})
    return out


def collect_robotwin(cfg: RTWX0FConfig, *, train: bool, rng: np.random.Generator) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env

    repo = Path(cfg.robotwin_repo)
    args = _load_task_args(repo, "demo_clean")
    args["task_name"] = TASK
    n_ep = cfg.n_train_ep if train else cfg.n_test_ep
    seed0 = cfg.seed if train else cfg.seed + 9000
    rows = {k: [] for k in ("tau_cmd", "qf_post_set", "passive", "qf_post", "ext", "qacc_pre", "qacc_post", "qdd_fd")}
    ext_api = False
    dt = float(cfg.dt)
    for i in range(n_ep):
        if i % 3 == 0 or i + 1 == n_ep:
            print(f"[rtwx-x0f] {TASK} {'train' if train else 'test'} ep {i+1}/{n_ep}", flush=True)
        env = _setup_env(repo, TASK, seed0 + i * 19, cfg.seed_attempts, args)
        art = _cabinet_art(env)
        phi = _phi_train(rng) if train else _phi_test()
        _prep_cabinet(env, phi)
        n = int(np.asarray(art.get_qpos()).reshape(-1).size)
        u = _random_tau(rng, cfg.n_steps, n)
        for t in range(cfg.n_steps):
            qd_pre = _vec(art, "get_qvel", n)
            rows["qacc_pre"].append(_vec(art, "get_qacc", n))
            tau = np.zeros(n)
            tau[: min(n, u[t].size)] = u[t][:n]
            rows["tau_cmd"].append(tau.copy())
            pas = _passive(art, n)
            rows["passive"].append(pas)
            ext, ok = _ext(art, n)
            ext_api = ext_api or ok
            rows["ext"].append(ext)
            qf = np.zeros(n)
            qf[:n] = tau
            art.set_qf(qf)
            rows["qf_post_set"].append(_vec(art, "get_qf", n))
            env.scene.step()
            qd_post = _vec(art, "get_qvel", n)
            rows["qacc_post"].append(_vec(art, "get_qacc", n))
            rows["qf_post"].append(_vec(art, "get_qf", n))
            rows["qdd_fd"].append((qd_post - qd_pre) / max(dt, 1e-8))
        try:
            env.close()
        except Exception:
            pass
    out = {k: np.stack(v) for k, v in rows.items()}
    out.update({"n_steps": int(cfg.n_steps), "n_ep": int(n_ep), "ext_api": bool(ext_api), "source": "sapien"})
    return out


def _channel_stack(pool: dict[str, Any]) -> dict[str, np.ndarray]:
    cmd = pool["tau_cmd"]
    pas = pool["passive"]
    ext = pool["ext"]
    return {
        "tau_cmd": cmd,
        "qf_post_set": pool["qf_post_set"],
        "cmd_plus_passive": cmd + pas,
        "qf_post": pool["qf_post"],
        "cmd_plus_passive_ext": cmd + pas + ext,
    }


def score_grid(pool: dict[str, Any], rho_min: float) -> dict[str, Any]:
    chans = _channel_stack(pool)
    n_steps, n_ep = int(pool["n_steps"]), int(pool["n_ep"])
    table = []
    best = {"score": -1.0}
    for k, tau in chans.items():
        for src in QDD_SRCS:
            qdd = pool[src]
            for lag in LAGS:
                sc = _min_dof_corr(tau, qdd, lag, n_steps, n_ep)
                rec = {"channel": k, "qdd": src, "lag": int(lag), "min_abs_corr": sc}
                table.append(rec)
                if sc > best["score"]:
                    best = {"score": sc, **rec}
    g0 = bool(best.get("score", 0) >= rho_min)
    return {"best": best, "grid": table, "G0_split": g0}


def run_rtwx_x0f(output: str | Path, *, config: RTWX0FConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0FConfig()
    root = Path(output).resolve()
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("r10_c0 is locked; RTWX-X0F must not write there")
    root.mkdir(parents=True, exist_ok=True)
    header = _write_header(root, cfg)
    rng = np.random.default_rng(cfg.seed)
    if cfg.backend == "numpy":
        train, test = collect_numpy(cfg, train=True, rng=rng), collect_numpy(cfg, train=False, rng=rng)
    else:
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        _patch_curobo_planner(repo)
        train = collect_robotwin(cfg, train=True, rng=rng)
        test = collect_robotwin(cfg, train=False, rng=rng)
    tr = score_grid(train, cfg.rho_min)
    te = score_grid(test, cfg.rho_min)
    star = tr["best"]
    # held-out: same (k, lag, qdd src)
    hold = next(
        (
            g
            for g in te["grid"]
            if g["channel"] == star.get("channel") and g["lag"] == star.get("lag") and g["qdd"] == star.get("qdd")
        ),
        {"min_abs_corr": 0.0},
    )
    g0 = bool(star.get("score", 0) >= cfg.rho_min and hold["min_abs_corr"] >= cfg.rho_min)
    pattern = "channel_reproduced" if g0 else "force_channel_unresolved"
    # G1 oracle closure is locked until G0; not run in this cell if G0 fails.
    # G1 skipped unless G0: no phi fit; pinocchio optional later
    summary = {
        "stage": "RTWX-X0F",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "header": header,
        "capacity_claim": False,
        "phi_fit": False,
        "unlocks_r10_c0": False,
        "G0": g0,
        "G1_oracle": None,
        "pattern": pattern,
        "rtwx_x0f_passed": g0,
        "train_best": star,
        "heldout_same_triple": hold,
        "train_top5": sorted(tr["grid"], key=lambda r: -r["min_abs_corr"])[:5],
        "test_top5": sorted(te["grid"], key=lambda r: -r["min_abs_corr"])[:5],
        "ext_api": bool(train.get("ext_api")),
        "source": train.get("source"),
        "n_train": int(train["n_ep"]),
        "n_test": int(test["n_ep"]),
        "d_q": int(train["tau_cmd"].shape[1]),
    }
    np.savez_compressed(
        root / "logs.npz",
        tau_cmd=train["tau_cmd"],
        qf_post_set=train["qf_post_set"],
        passive=train["passive"],
        qf_post=train["qf_post"],
        ext=train["ext"],
        qacc_pre=train["qacc_pre"],
        qacc_post=train["qacc_post"],
        qdd_fd=train["qdd_fd"],
    )
    _write_json(root / "summary.json", summary)
    _write_json(root / "grid_train.json", {"grid": tr["grid"]})
    return summary
