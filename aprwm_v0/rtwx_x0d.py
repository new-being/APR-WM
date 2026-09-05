"""RTWX-X0D: native closed-loop structure on (q, qd, qtar) → (q', qd'). No force, no nets."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0c import (
    FORMAL_N_STEPS,
    FORMAL_N_TEST,
    FORMAL_N_TRAIN,
    FORMAL_N_VAL,
    collect_robotwin,
    _jsonable,
    _lock_robotwin_scale,
    _write_json,
)
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0D_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0d.closed_loop_structure.v1"
E1_ID_MIN = 0.02
DQ_STD_MIN = 1.0e-4
G1_RATIO = 0.75
G3_RATIO = 1.10
ROLL50_BLOW_MULT = 10.0
SEED_FORMAL = 8501


@dataclass(frozen=True)
class RTWX0DConfig:
    output: str = "runs/rtwx_x0d"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED_FORMAL
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0D must not write there")


def _write_header(root: Path, cfg: RTWX0DConfig) -> dict[str, Any]:
    header = {
        "stage": "RTWX-X0D",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_before_score": True,
        "primary": "(q,qd) one-step and rollout; not qdd; not tau",
        "vartheta": "effective closed-loop system coordinates (not mass/inertia/torque gain)",
        "capacity_claim": False,
        "neural": False,
        "unlocks_x0d1_only_if_pass": True,
        "gates_frozen": {
            "E1_identity_min": E1_ID_MIN,
            "dq_std_min": DQ_STD_MIN,
            "G1_ratio": G1_RATIO,
            "G3_ratio": G3_RATIO,
            "roll50_blow_mult": ROLL50_BLOW_MULT,
        },
        "scale": {
            "n_train_ep": cfg.n_train_ep,
            "n_val_ep": cfg.n_val_ep,
            "n_test_ep": cfg.n_test_ep,
            "n_steps": cfg.n_steps,
            "backend": cfg.backend,
            "smoke": cfg.smoke,
            "seed": cfg.seed,
        },
        "formal_robotwin_frozen": {
            "n_train_ep": FORMAL_N_TRAIN,
            "n_val_ep": FORMAL_N_VAL,
            "n_test_ep": FORMAL_N_TEST,
            "n_steps": FORMAL_N_STEPS,
        },
        "note": "oracle simulator state / native-qpos closed-loop structure; not official observation benchmark",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0D header (before score)\n"
        f"backend={cfg.backend} smoke={cfg.smoke}\n"
        f"n_train_ep={cfg.n_train_ep} n_val_ep={cfg.n_val_ep} n_test_ep={cfg.n_test_ep} n_steps={cfg.n_steps}\n"
        f"E1_identity_min={E1_ID_MIN} G1_ratio={G1_RATIO} G3_ratio={G3_RATIO}\n"
        "primary=(q,qd) capacity_claim=false neural=false no_tau\n",
        encoding="utf-8",
    )
    return header


def _x(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd], axis=-1)


def _e1(qhat: np.ndarray, qdhat: np.ndarray, qn: np.ndarray, qdn: np.ndarray) -> dict[str, float]:
    xh = _x(qhat, qdhat)
    xr = _x(qn, qdn)
    return {
        "E_1": _nrmse(xh, xr),
        "E_1_q": _nrmse(qhat, qn),
        "E_1_qd": _nrmse(qdhat, qdn),
    }


def _lstsq_map(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y (N,d) ~ phi (N,p) W; W is (p,d)."""
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return np.asarray(w, dtype=np.float64)


def _phi_m1(qd: np.ndarray, eq: np.ndarray, j: int) -> np.ndarray:
    return np.column_stack([qd[:, j], eq[:, j], np.ones(qd.shape[0])])


def _fit_m1(qd: np.ndarray, eq: np.ndarray, qdn: np.ndarray) -> np.ndarray:
    d = qd.shape[1]
    coef = np.zeros((d, 3), dtype=np.float64)
    for j in range(d):
        coef[j] = _lstsq_map(_phi_m1(qd, eq, j), qdn[:, j : j + 1]).reshape(-1)
    return coef


def _pred_m1_qd(qd: np.ndarray, eq: np.ndarray, coef: np.ndarray) -> np.ndarray:
    out = np.zeros_like(qd)
    for j in range(qd.shape[1]):
        out[:, j] = _phi_m1(qd, eq, j) @ coef[j]
    return out


def _phi_m2(qd: np.ndarray, eq: np.ndarray) -> np.ndarray:
    return np.column_stack([qd, eq, np.ones((qd.shape[0], 1))])


def _phi_m3(qd: np.ndarray, eq: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.column_stack([qd, eq, np.sin(q), np.cos(q), np.ones((qd.shape[0], 1))])


def _pred_q_euler(q: np.ndarray, qd: np.ndarray, dt: np.ndarray) -> np.ndarray:
    return q + dt.reshape(-1, 1) * qd


def _step_m0(q: np.ndarray, qd: np.ndarray, eq: np.ndarray, dt: float, pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return q.copy(), qd.copy()


def _step_m1(q: np.ndarray, qd: np.ndarray, eq: np.ndarray, dt: float, pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coef = pack["m1"]
    qn = q + dt * qd
    qdn = np.zeros_like(qd)
    for j in range(qd.size):
        qdn[j] = coef[j, 0] * qd[j] + coef[j, 1] * eq[j] + coef[j, 2]
    return qn, qdn


def _step_lin(q: np.ndarray, qd: np.ndarray, eq: np.ndarray, dt: float, pack: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    w = pack[key]
    qn = q + dt * qd
    if key == "m2":
        phi = np.concatenate([qd, eq, np.ones(1)])
    else:
        phi = np.concatenate([qd, eq, np.sin(q), np.cos(q), np.ones(1)])
    qdn = phi @ w
    return qn, qdn


def _rollout(pool: dict[str, Any], step_fn: Callable[..., tuple[np.ndarray, np.ndarray]], pack: dict[str, Any], horizon: int) -> float:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    if n_ep == 0 or n_steps <= horizon:
        return float("inf")
    q, qd, qn, qdn = pool["q"], pool["qd"], pool["qn"], pool["qdn"]
    qtar, dt = pool["q_tar"], pool["dt"]
    errs: list[float] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt, dts = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl], dt[sl]
        stride = max(1, horizon)
        for t0 in range(0, n_steps - horizon, stride):
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            pred, ref = [], []
            for h in range(horizon):
                eq = qt[t0 + h] - qh
                qh, qdh = step_fn(qh, qdh, eq, float(dts[t0 + h]), pack)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    return float("inf")
                pred.append(_x(qh, qdh))
                ref.append(_x(qnn[t0 + h], qdnn[t0 + h]))
            errs.append(_nrmse(np.stack(pred), np.stack(ref)))
    return float(np.mean(errs)) if errs else float("inf")


def _eval_split(pool: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    q, qd, qn, qdn, eq, dt = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["eq"], pool["dt"]
    p0q, p0d = q, qd
    p1q = _pred_q_euler(q, qd, dt)
    p1d = _pred_m1_qd(qd, eq, pack["m1"])
    p2q = p1q
    p2d = _phi_m2(qd, eq) @ pack["m2"]
    p3q = p1q
    p3d = _phi_m3(qd, eq, q) @ pack["m3"]
    out: dict[str, Any] = {}
    for name, hq, hd, fn in (
        ("M0", p0q, p0d, _step_m0),
        ("M1", p1q, p1d, _step_m1),
        ("M2", p2q, p2d, lambda a, b, c, d, p: _step_lin(a, b, c, d, p, "m2")),
        ("M3", p3q, p3d, lambda a, b, c, d, p: _step_lin(a, b, c, d, p, "m3")),
    ):
        blk = _e1(hq, hd, qn, qdn)
        blk["E_roll10"] = _rollout(pool, fn, pack, 10)
        blk["E_roll50"] = _rollout(pool, fn, pack, 50)
        out[name] = blk
    return out


def _g1_ok(e_struct: float, e_id: float) -> bool:
    if not np.isfinite(e_struct) or not np.isfinite(e_id) or e_id <= 0:
        return False
    return bool(e_struct <= G1_RATIO * e_id)


def _g2_ok(struct: dict[str, Any], ident: dict[str, Any]) -> bool:
    r10s, r10i = struct["E_roll10"], ident["E_roll10"]
    r50s, r50i = struct["E_roll50"], ident["E_roll50"]
    if not np.isfinite(r10s) or not np.isfinite(r10i):
        return False
    if not (r10s < r10i):
        return False
    if not np.isfinite(r50s):
        return False
    cap = ROLL50_BLOW_MULT * max(float(r50i) if np.isfinite(r50i) else 0.0, 1.0e-8)
    return bool(r50s < cap)


def _score(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    eq = train["q_tar"] - train["q"]
    pack = {
        "m1": _fit_m1(train["qd"], eq, train["qdn"]),
        "m2": _lstsq_map(_phi_m2(train["qd"], eq), train["qdn"]),
        "m3": _lstsq_map(_phi_m3(train["qd"], eq, train["q"]), train["qdn"]),
    }
    tr, va, te = _eval_split(train, pack), _eval_split(val, pack), _eval_split(test, pack)

    def g0_split(p: dict[str, Any], blk: dict[str, Any]) -> dict[str, bool]:
        dq = p["qn"] - p["q"]
        return {
            "E1_id": bool(blk["M0"]["E_1"] >= E1_ID_MIN),
            "dq_std": bool(float(np.mean(np.std(dq, axis=0))) >= DQ_STD_MIN),
        }

    g0_bits = {
        "train": g0_split(train, tr),
        "val": g0_split(val, va),
        "test": g0_split(test, te),
    }
    g0 = all(all(v.values()) for v in g0_bits.values()) and int(train["n_ep"]) > 0 and int(val["n_ep"]) > 0 and int(test["n_ep"]) > 0

    names = ("M1", "M2", "M3")
    val_g1 = [k for k in names if _g1_ok(va[k]["E_1"], va["M0"]["E_1"])]
    test_g1 = [k for k in names if _g1_ok(te[k]["E_1"], te["M0"]["E_1"])]
    g1 = bool(len(test_g1) > 0)
    kstar = None
    if val_g1:
        kstar = min(val_g1, key=lambda k: va[k]["E_1"])
    g2 = bool(kstar is not None and _g2_ok(te[kstar], te["M0"]))
    g3 = False
    if kstar is not None:
        g3 = bool(np.isfinite(te[kstar]["E_1"]) and np.isfinite(va[kstar]["E_1"]) and te[kstar]["E_1"] <= G3_RATIO * va[kstar]["E_1"])

    if not g0:
        pattern = "instrument_failure"
    elif g0 and g1 and g2 and g3:
        only_m3 = test_g1 == ["M3"]
        pattern = "configuration_structure_required" if only_m3 else "closed_loop_structure_supported"
    else:
        pattern = "low_order_structure_insufficient"

    passed = pattern in {"closed_loop_structure_supported", "configuration_structure_required"}
    return {
        "G0": g0,
        "G0_bits": g0_bits,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "kstar": kstar,
        "val_g1_models": val_g1,
        "test_g1_models": test_g1,
        "pattern": pattern,
        "rtwx_x0d_passed": passed,
        "unlocks_x0d1": passed,
        "train": tr,
        "val": va,
        "test": te,
        "m1_coef": pack["m1"].tolist(),
    }


def collect_numpy_x0d(cfg: RTWX0DConfig, *, split: str, rng: np.random.Generator) -> dict[str, Any]:
    """Discrete closed-loop plant matching M1 (explicit Euler + per-joint affine qd)."""
    from .rtwx_x0c import _multisine

    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    dt = float(cfg.dt_numpy)
    n_dof = int(cfg.n_dof_numpy)
    base = np.array([0.72, 0.68, 0.75, 0.70], dtype=np.float64)
    a = np.resize(base, n_dof)
    b = 0.35 * np.ones(n_dof)
    c = np.zeros(n_dof)
    lo = -1.2 * np.ones(n_dof)
    hi = 1.2 * np.ones(n_dof)
    t = np.arange(n_steps, dtype=np.float64) * dt
    eps: list[dict[str, np.ndarray]] = []
    for _ in range(n_ep):
        q = rng.uniform(-0.2, 0.2, size=n_dof)
        qd = rng.uniform(-0.05, 0.05, size=n_dof)
        qtar, _qdt = _multisine(q, lo, hi, t, rng)
        rows: dict[str, list] = {k: [] for k in ("q", "qd", "qn", "qdn", "q_tar", "dt")}
        for i in range(n_steps):
            eq = qtar[i] - q
            qn = q + dt * qd
            qdn = a * qd + b * eq + c
            rows["q"].append(q.copy())
            rows["qd"].append(qd.copy())
            rows["qn"].append(qn.copy())
            rows["qdn"].append(qdn.copy())
            rows["q_tar"].append(qtar[i].copy())
            rows["dt"].append(dt)
            q, qd = qn, qdn
        ep = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items() if k != "dt"}
        ep["dt"] = np.asarray(rows["dt"], dtype=np.float64)
        eps.append(ep)
    out: dict[str, Any] = {
        "q": np.concatenate([e["q"] for e in eps], axis=0),
        "qd": np.concatenate([e["qd"] for e in eps], axis=0),
        "qn": np.concatenate([e["qn"] for e in eps], axis=0),
        "qdn": np.concatenate([e["qdn"] for e in eps], axis=0),
        "q_tar": np.concatenate([e["q_tar"] for e in eps], axis=0),
        "dt": np.concatenate([e["dt"] for e in eps], axis=0),
        "n_ep": len(eps),
        "n_steps": n_steps,
        "n_dof": n_dof,
        "source": "numpy",
    }
    out["eq"] = out["q_tar"] - out["q"]
    return out


def run_rtwx_x0d(output: str | Path, config: RTWX0DConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWX0DConfig(output=str(output))
    cfg = _lock_robotwin_scale(cfg)  # type: ignore[arg-type]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = _write_header(root, cfg)
    stop_mode: list[str] = []
    rng_tr = np.random.default_rng(cfg.seed)
    rng_va = np.random.default_rng(cfg.seed + 1)
    rng_te = np.random.default_rng(cfg.seed + 2)
    if cfg.backend == "numpy":
        train = collect_numpy_x0d(cfg, split="train", rng=rng_tr)
        val = collect_numpy_x0d(cfg, split="val", rng=rng_va)
        test = collect_numpy_x0d(cfg, split="test", rng=rng_te)
    elif cfg.backend == "robotwin":
        repo = Path(cfg.robotwin_repo)
        os.chdir(repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
        _patch_curobo_planner(repo)
        train = collect_robotwin(cfg, split="train", rng=rng_tr, stop_mode=stop_mode)  # type: ignore[arg-type]
        if stop_mode:
            scored = {
                "G0": False,
                "G1": False,
                "G2": False,
                "G3": False,
                "pattern": "instrument_failure",
                "rtwx_x0d_passed": False,
                "unlocks_x0d1": False,
                "kstar": None,
            }
            summary = {"header": header, "capacity_claim": False, "neural": False, **scored}
            _write_json(root / "summary.json", summary)
            _write_json(root / "run.json", summary)
            return summary
        val = collect_robotwin(cfg, split="val", rng=rng_va, stop_mode=stop_mode)  # type: ignore[arg-type]
        test = collect_robotwin(cfg, split="test", rng=rng_te, stop_mode=stop_mode)  # type: ignore[arg-type]
    else:
        raise ValueError(cfg.backend)

    for p in (train, val, test):
        if "qn" not in p or p["qn"].shape[0] == 0:
            raise RuntimeError("X0D requires qn/qdn from collector")
        p["eq"] = p["q_tar"] - p["q"]

    scored = _score(train, val, test)
    summary = {
        "header": header,
        "capacity_claim": False,
        "neural": False,
        "label": "oracle simulator state / native-qpos closed-loop structure",
        "not_official_robotwin_observation_benchmark": True,
        "not_force_level_id": True,
        "n_train": int(train.get("n_ep", 0)),
        "n_val": int(val.get("n_ep", 0)),
        "n_test": int(test.get("n_ep", 0)),
        "n_steps": int(train.get("n_steps", 0)),
        "d_arm": int(train.get("n_dof", 0)),
        "source": train.get("source"),
        **scored,
    }
    np.savez_compressed(
        root / "logs.npz",
        q=train["q"],
        qd=train["qd"],
        qn=train["qn"],
        qdn=train["qdn"],
        q_tar=train["q_tar"],
        dt=train["dt"],
    )
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "G0": scored["G0"],
            "G1": scored["G1"],
            "G2": scored["G2"],
            "G3": scored["G3"],
            "pattern": scored["pattern"],
            "kstar": scored["kstar"],
            "test": scored["test"],
            "val": scored["val"],
        },
    )
    return summary
