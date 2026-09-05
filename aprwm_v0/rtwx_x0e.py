"""RTWX-X0E: direct native-window Δstate. New family (not X0D patch). No Euler, no force, no nets."""

from __future__ import annotations

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
    _write_json,
)
from .rtwx_x0_smoke import _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/RTWX0E_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0e.direct_increment.v1"
E1_ID_MIN = 0.02
DQ_STD_MIN = 1.0e-4
G1_RATIO = 0.75
G3_RATIO = 1.10
ROLL50_BLOW_MULT = 10.0
SEED_FORMAL = 8601


@dataclass(frozen=True)
class RTWX0EConfig:
    output: str = "runs/rtwx_x0e"
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
        raise RuntimeError("r10_c0 is locked; RTWX-X0E must not write there")


def _lock(cfg: RTWX0EConfig) -> RTWX0EConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
    )


def _write_header(root: Path, cfg: RTWX0EConfig) -> dict[str, Any]:
    header = {
        "stage": "RTWX-X0E",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "frozen_before_score": True,
        "family": "direct native-window increment (not X0D patch; not physics closure)",
        "forbids": ["euler_q_update", "set_qf", "get_qf", "mlp", "diffusion", "neural_residual"],
        "primary": "E_1 and rollout on (q,qd); predict Delta x, not Delta t * qd",
        "vartheta": "native-window increment coordinates (not mass/inertia/torque gain)",
        "capacity_claim": False,
        "neural": False,
        "unlocks_x0e1_only_if_pass": True,
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
        "note": "oracle simulator state / native action window increment; not official observation benchmark",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    (root / "run.header.txt").write_text(
        "RTWX-X0E header (before score)\n"
        f"backend={cfg.backend} smoke={cfg.smoke}\n"
        f"n_train_ep={cfg.n_train_ep} n_val_ep={cfg.n_val_ep} n_test_ep={cfg.n_test_ep} n_steps={cfg.n_steps}\n"
        f"E1_identity_min={E1_ID_MIN} G1_ratio={G1_RATIO} G3_ratio={G3_RATIO}\n"
        "primary=Delta_x capacity_claim=false neural=false no_euler\n",
        encoding="utf-8",
    )
    return header


def _x(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    return np.concatenate([q, qd], axis=-1)


def _e1(qhat: np.ndarray, qdhat: np.ndarray, qn: np.ndarray, qdn: np.ndarray) -> dict[str, float]:
    return {
        "E_1": _nrmse(_x(qhat, qdhat), _x(qn, qdn)),
        "E_1_q": _nrmse(qhat, qn),
        "E_1_qd": _nrmse(qdhat, qdn),
    }


def _lstsq(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return np.asarray(w, dtype=np.float64)


def _phi_m1(q: np.ndarray, qd: np.ndarray, eq: np.ndarray) -> np.ndarray:
    return np.column_stack([q, qd, eq, np.ones((q.shape[0], 1))])


def _phi_m2(q: np.ndarray, qd: np.ndarray, eq: np.ndarray) -> np.ndarray:
    return np.column_stack([q, qd, eq, np.sin(q), np.cos(q), eq * np.abs(eq), np.ones((q.shape[0], 1))])


def _apply_dx(q: np.ndarray, qd: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = q.shape[-1]
    return q + dx[..., :d], qd + dx[..., d:]


def _step_m0(q: np.ndarray, qd: np.ndarray, eq: np.ndarray, pack: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return q.copy(), qd.copy()


def _step_lin(q: np.ndarray, qd: np.ndarray, eq: np.ndarray, pack: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    q2 = np.asarray(q, dtype=np.float64).reshape(1, -1)
    qd2 = np.asarray(qd, dtype=np.float64).reshape(1, -1)
    eq2 = np.asarray(eq, dtype=np.float64).reshape(1, -1)
    phi = _phi_m1(q2, qd2, eq2) if key == "m1" else _phi_m2(q2, qd2, eq2)
    dx = (phi @ pack[key]).reshape(-1)
    return _apply_dx(np.asarray(q, dtype=np.float64).reshape(-1), np.asarray(qd, dtype=np.float64).reshape(-1), dx)


def _rollout(pool: dict[str, Any], step_fn: Callable[..., tuple[np.ndarray, np.ndarray]], pack: dict[str, Any], horizon: int) -> float:
    n_ep = int(pool["n_ep"])
    n_steps = int(pool["n_steps"])
    if n_ep == 0 or n_steps <= horizon:
        return float("inf")
    q, qd, qn, qdn = pool["q"], pool["qd"], pool["qn"], pool["qdn"]
    qtar = pool["q_tar"]
    errs: list[float] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        stride = max(1, horizon)
        for t0 in range(0, n_steps - horizon, stride):
            qh, qdh = qq[t0].copy(), qddt[t0].copy()
            pred, ref = [], []
            for h in range(horizon):
                eq = qt[t0 + h] - qh
                qh, qdh = step_fn(qh, qdh, eq, pack)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    return float("inf")
                pred.append(_x(qh, qdh))
                ref.append(_x(qnn[t0 + h], qdnn[t0 + h]))
            errs.append(_nrmse(np.stack(pred), np.stack(ref)))
    return float(np.mean(errs)) if errs else float("inf")


def _eval_split(pool: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    q, qd, qn, qdn, eq = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["eq"]
    dx1 = _phi_m1(q, qd, eq) @ pack["m1"]
    dx2 = _phi_m2(q, qd, eq) @ pack["m2"]
    p1q, p1d = _apply_dx(q, qd, dx1)
    p2q, p2d = _apply_dx(q, qd, dx2)
    out: dict[str, Any] = {}
    for name, hq, hd, fn in (
        ("M0", q, qd, _step_m0),
        ("M1", p1q, p1d, lambda a, b, c, p: _step_lin(a, b, c, p, "m1")),
        ("M2", p2q, p2d, lambda a, b, c, p: _step_lin(a, b, c, p, "m2")),
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
    if not np.isfinite(r10s) or not np.isfinite(r10i) or not (r10s < r10i):
        return False
    if not np.isfinite(r50s):
        return False
    cap = ROLL50_BLOW_MULT * max(float(r50i) if np.isfinite(r50i) else 0.0, 1.0e-8)
    return bool(r50s < cap)


def _score(train: dict[str, Any], val: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    dq = train["qn"] - train["q"]
    dqd = train["qdn"] - train["qd"]
    y = np.concatenate([dq, dqd], axis=1)
    pack = {
        "m1": _lstsq(_phi_m1(train["q"], train["qd"], train["eq"]), y),
        "m2": _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), y),
    }
    tr, va, te = _eval_split(train, pack), _eval_split(val, pack), _eval_split(test, pack)

    def g0_split(p: dict[str, Any], blk: dict[str, Any]) -> dict[str, bool]:
        dqq = p["qn"] - p["q"]
        return {
            "E1_id": bool(blk["M0"]["E_1"] >= E1_ID_MIN),
            "dq_std": bool(float(np.mean(np.std(dqq, axis=0))) >= DQ_STD_MIN),
        }

    g0_bits = {"train": g0_split(train, tr), "val": g0_split(val, va), "test": g0_split(test, te)}
    g0 = all(all(v.values()) for v in g0_bits.values()) and int(train["n_ep"]) > 0

    names = ("M1", "M2")
    val_g1 = [k for k in names if _g1_ok(va[k]["E_1"], va["M0"]["E_1"])]
    test_g1 = [k for k in names if _g1_ok(te[k]["E_1"], te["M0"]["E_1"])]
    g1 = bool(len(test_g1) > 0)
    kstar = min(val_g1, key=lambda k: va[k]["E_1"]) if val_g1 else None
    g2 = bool(kstar is not None and _g2_ok(te[kstar], te["M0"]))
    g3 = False
    if kstar is not None:
        g3 = bool(
            np.isfinite(te[kstar]["E_1"])
            and np.isfinite(va[kstar]["E_1"])
            and te[kstar]["E_1"] <= G3_RATIO * va[kstar]["E_1"]
        )

    if not g0:
        pattern = "instrument_failure"
    elif g0 and g1 and g2 and g3:
        if "M1" in test_g1:
            pattern = "native_window_increment_supported"
        else:
            pattern = "nonlinear_increment_required"
    else:
        pattern = "low_capacity_increment_insufficient"

    passed = pattern in {"native_window_increment_supported", "nonlinear_increment_required"}
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
        "rtwx_x0e_passed": passed,
        "unlocks_x0e1": passed,
        "train": tr,
        "val": va,
        "test": te,
        "m1_shape": list(pack["m1"].shape),
        "m2_shape": list(pack["m2"].shape),
    }


def collect_numpy_x0e(cfg: RTWX0EConfig, *, split: str, rng: np.random.Generator) -> dict[str, Any]:
    """True linear increment plant: Delta x = W [q,qd,eq] + b (no Euler)."""
    from .rtwx_x0c import _multisine

    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    n_dof = int(cfg.n_dof_numpy)
    dt = float(cfg.dt_numpy)
    lo = -1.2 * np.ones(n_dof)
    hi = 1.2 * np.ones(n_dof)
    t = np.arange(n_steps, dtype=np.float64) * dt
    a_q, b_q = 0.12, 0.45
    a_d, b_d = 0.08, 0.30
    eps: list[dict[str, np.ndarray]] = []
    for _ in range(n_ep):
        q = rng.uniform(-0.2, 0.2, size=n_dof)
        qd = rng.uniform(-0.05, 0.05, size=n_dof)
        qtar, _ = _multisine(q, lo, hi, t, rng)
        rows: dict[str, list] = {k: [] for k in ("q", "qd", "qn", "qdn", "q_tar")}
        for i in range(n_steps):
            eq = qtar[i] - q
            dq = a_q * qd + b_q * eq
            dqd = -a_d * qd + b_d * eq
            qn, qdn = q + dq, qd + dqd
            rows["q"].append(q.copy())
            rows["qd"].append(qd.copy())
            rows["qn"].append(qn.copy())
            rows["qdn"].append(qdn.copy())
            rows["q_tar"].append(qtar[i].copy())
            q, qd = qn, qdn
        ep = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
        eps.append(ep)
    out: dict[str, Any] = {
        "q": np.concatenate([e["q"] for e in eps], axis=0),
        "qd": np.concatenate([e["qd"] for e in eps], axis=0),
        "qn": np.concatenate([e["qn"] for e in eps], axis=0),
        "qdn": np.concatenate([e["qdn"] for e in eps], axis=0),
        "q_tar": np.concatenate([e["q_tar"] for e in eps], axis=0),
        "n_ep": len(eps),
        "n_steps": n_steps,
        "n_dof": n_dof,
        "source": "numpy",
    }
    out["eq"] = out["q_tar"] - out["q"]
    return out


def run_rtwx_x0e(output: str | Path, config: RTWX0EConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0EConfig(output=str(output)))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root.resolve())
    header = _write_header(root, cfg)
    stop_mode: list[str] = []
    rng_tr = np.random.default_rng(cfg.seed)
    rng_va = np.random.default_rng(cfg.seed + 1)
    rng_te = np.random.default_rng(cfg.seed + 2)
    if cfg.backend == "numpy":
        train = collect_numpy_x0e(cfg, split="train", rng=rng_tr)
        val = collect_numpy_x0e(cfg, split="val", rng=rng_va)
        test = collect_numpy_x0e(cfg, split="test", rng=rng_te)
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
                "rtwx_x0e_passed": False,
                "unlocks_x0e1": False,
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
        if p["qn"].shape[0] == 0:
            raise RuntimeError("X0E requires qn/qdn from collector")
        p["eq"] = p["q_tar"] - p["q"]

    scored = _score(train, val, test)
    summary = {
        "header": header,
        "capacity_claim": False,
        "neural": False,
        "label": "oracle simulator state / native-window direct increment",
        "not_official_robotwin_observation_benchmark": True,
        "not_force_level_id": True,
        "not_x0d_patch": True,
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
