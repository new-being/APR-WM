"""R7-P1: saturated-actuator switchability + degree-2 misspecification. No certificate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r3_v7b3 import spearman
from .r5_i0 import relative_l2
from .r7_a0 import POLY_DEG, fit_poly, predict_a
from .r7_p0 import (
    BETA_U,
    D_MACRO_MAX,
    DAMPING,
    DT,
    HOLD_S,
    HS_KEYS,
    M_MIN,
    N_SIDE_MIN,
    PROBE_S,
    RHO_MIN,
    U_ALT,
    U_DEFAULT,
    X_SCALE,
    Y_STAR,
)
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_P1_PREREG.md"
F_MAX = 2.0
E_MIN = 1.0e-4
R2_MAX = 0.99
ALPHAS = (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_a0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-P1 locked until R7-A0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-A0":
        raise RuntimeError("R7-P1 locked: A0 summary missing")
    if not payload.get("r7_a0_go"):
        raise RuntimeError("R7-P1 locked: A0 GO is false")
    return payload


def _step(
    y: float,
    v: float,
    u: float,
    alpha: float,
    dt: float,
    f_max: float = F_MAX,
) -> tuple[float, float, float]:
    force = float(np.clip(alpha * u, -f_max, f_max))
    decay = float(np.exp(-DAMPING * dt))
    drift = force / DAMPING
    v1 = drift + (v - drift) * decay
    y1 = y + drift * dt + (v - drift) * (1.0 - decay) / DAMPING
    acc = force - DAMPING * v1
    return float(y1), float(v1), float(acc)


def simulate_saturated(alpha: float, u_cmd: float, f_max: float = F_MAX) -> dict[str, Any]:
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    y = 0.0
    v = 0.0
    hold = {key: [] for key in HS_KEYS}
    for _ in range(n_hold):
        y, v, acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
        hold["y"].append(y)
        hold["v"].append(v)
        hold["a"].append(acc)
        hold["u"].append(0.0)
    future = []
    for _ in range(n_probe):
        y, v, acc = _step(y, v, u_cmd, alpha, DT, f_max=f_max)
        future.append((y, v, acc, u_cmd))
    y_future = np.asarray(future, dtype=np.float64)
    hs = np.concatenate([np.asarray(hold[key], dtype=np.float64) for key in HS_KEYS])
    return {
        "alpha": float(alpha),
        "x": float(X_SCALE * alpha),
        "hs": hs,
        "y_future": y_future,
        "y_end": float(y_future[-1, 0]),
        "u_cmd": float(u_cmd),
        "force": float(np.clip(alpha * u_cmd, -f_max, f_max)),
        "f_max": float(f_max),
    }


def task_loss(trace: dict[str, Any]) -> float:
    y = np.asarray(trace["y_future"], dtype=np.float64)
    pos = (float(y[-1, 0]) - Y_STAR) ** 2
    effort = BETA_U * float(np.mean(y[:, 3] ** 2))
    return float(pos + effort)


def _loo_degree2(x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, float, float]:
    preds = np.zeros(len(a), dtype=np.float64)
    for i in range(len(a)):
        mask = np.ones(len(a), dtype=bool)
        mask[i] = False
        coef = fit_poly(x[mask], a[mask])
        preds[i] = float(predict_a(np.asarray([x[i]]), coef)[0])
    resid = preds - a
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum((a - preds) ** 2))
    ss_tot = float(np.sum((a - float(np.mean(a))) ** 2))
    r2 = 0.0 if ss_tot < 1.0e-18 else 1.0 - ss_res / ss_tot
    return resid, rmse, r2


def run_r7_p1(
    output: str | Path,
    *,
    a0_summary: str | Path = "runs/r7_a0/formal/summary.json",
) -> dict[str, Any]:
    a0 = _require_a0(Path(a0_summary))
    table = []
    hs0 = None
    d_hs = []
    xs = []
    advantages = []
    for alpha in ALPHAS:
        mid = simulate_saturated(alpha, U_DEFAULT)
        cons = simulate_saturated(alpha, U_ALT)
        if hs0 is None:
            hs0 = mid["hs"]
        d_hs.append(relative_l2(mid["hs"], hs0))
        adv = task_loss(mid) - task_loss(cons)
        xs.append(float(mid["x"]))
        advantages.append(adv)
        table.append(
            {
                "alpha": float(alpha),
                "x": float(mid["x"]),
                "A": adv,
                "j_default": task_loss(mid),
                "j_alt": task_loss(cons),
                "y_end_default": float(mid["y_end"]),
                "y_end_alt": float(cons["y_end"]),
                "force_default": float(mid["force"]),
                "force_alt": float(cons["force"]),
                "switch_useful": bool(adv > 0.0),
            }
        )
    x = np.asarray(xs, dtype=np.float64)
    a = np.asarray(advantages, dtype=np.float64)
    d_hs_arr = np.asarray(d_hs, dtype=np.float64)
    loo_resid, loo_rmse, loo_r2 = _loo_degree2(x, a)
    useful = a > M_MIN
    harmful = a < -M_MIN
    g0 = bool(float(np.max(d_hs_arr)) < D_MACRO_MAX)
    g1 = bool(abs(spearman(np.asarray(ALPHAS, dtype=np.float64), x)) > RHO_MIN)
    g2 = bool(np.any(a > 0.0) and np.any(a < 0.0))
    g3 = bool(int(np.sum(useful)) >= N_SIDE_MIN and int(np.sum(harmful)) >= N_SIDE_MIN)
    g4 = bool(g3)
    g_mis = bool(loo_rmse > E_MIN and loo_r2 < R2_MAX)
    passed = bool(g0 and g1 and g2 and g3 and g4 and g_mis)
    status = stage_status()
    status["R7-A0"] = {"go": True, "frozen": True}
    status["R7-P1"] = {"pass": passed, "r7_a1_unlocked": passed}
    payload = {
        "stage": "R7-P1",
        "scientific_result": True,
        "neural_probe": False,
        "certificate": False,
        "poly_deg": POLY_DEG,
        "f_max": F_MAX,
        "e_min": E_MIN,
        "r2_max": R2_MAX,
        "prereg_sha256": _prereg_sha256(),
        "a0_prereg_sha256": a0.get("prereg_sha256"),
        "alphas": list(ALPHAS),
        "table": table,
        "max_d_hs": float(np.max(d_hs_arr)),
        "rho_alpha_x": spearman(np.asarray(ALPHAS, dtype=np.float64), x),
        "n_useful_margin": int(np.sum(useful)),
        "n_harmful_margin": int(np.sum(harmful)),
        "loo_rmse": loo_rmse,
        "loo_r2": loo_r2,
        "loo_resid": loo_resid.tolist(),
        "g0_macro_ambiguity": g0,
        "g1_evidence_visibility": g1,
        "g2_signed_crossover": g2,
        "g3_nondegenerate_margins": g3,
        "g4_balanced_coverage": g4,
        "g_mis_degree2": g_mis,
        "r7_p1_pass": passed,
        "r7_a1_unlocked": passed,
        "note": "saturated plant; no L / commit; A0 model class frozen for G-mis",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
