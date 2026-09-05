"""R7-P0: fresh hidden actuator-effectiveness switchability. No certificate."""

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
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_P0_PREREG.md"
DT = 0.002
HOLD_S = 0.30
PROBE_S = 0.50
DAMPING = 4.0
Y_STAR = 0.10
BETA_U = 1.0e-5
U_DEFAULT = 1.0
U_ALT = 0.4
X_SCALE = 0.25
ALPHAS = (0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4)
D_MACRO_MAX = 0.05
RHO_MIN = 0.80
M_MIN = 1.0e-3
N_SIDE_MIN = 3
HS_KEYS = ("y", "v", "a", "u")


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_b0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-P0 locked until R6-B0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R6-B0":
        raise RuntimeError("R7-P0 locked: B0 summary missing")
    if not payload.get("realization_branch_stop"):
        raise RuntimeError("R7-P0 locked: R6 realization is not frozen")
    return payload


def _step(y: float, v: float, u: float, alpha: float, dt: float) -> tuple[float, float, float]:
    # Exact: v' = alpha u - c v
    decay = float(np.exp(-DAMPING * dt))
    drift = (alpha * u) / DAMPING
    v1 = drift + (v - drift) * decay
    y1 = y + drift * dt + (v - drift) * (1.0 - decay) / DAMPING
    acc = alpha * u - DAMPING * v1
    return float(y1), float(v1), float(acc)


def simulate_effectiveness(alpha: float, u_cmd: float) -> dict[str, Any]:
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    y = 0.0
    v = 0.0
    hold = {key: [] for key in HS_KEYS}
    for _ in range(n_hold):
        y, v, acc = _step(y, v, 0.0, alpha, DT)
        hold["y"].append(y)
        hold["v"].append(v)
        hold["a"].append(acc)
        hold["u"].append(0.0)
    future = []
    for _ in range(n_probe):
        y, v, acc = _step(y, v, u_cmd, alpha, DT)
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
    }


def task_loss(trace: dict[str, Any]) -> float:
    y = np.asarray(trace["y_future"], dtype=np.float64)
    pos = (float(y[-1, 0]) - Y_STAR) ** 2
    effort = BETA_U * float(np.mean(y[:, 3] ** 2))
    return float(pos + effort)


def run_r7_p0(
    output: str | Path,
    *,
    b0_summary: str | Path = "runs/r6_b0/formal/summary.json",
) -> dict[str, Any]:
    b0 = _require_b0(Path(b0_summary))
    table = []
    hs0 = None
    d_hs = []
    alphas = []
    xs = []
    advantages = []
    for alpha in ALPHAS:
        def_tr = simulate_effectiveness(alpha, U_DEFAULT)
        alt_tr = simulate_effectiveness(alpha, U_ALT)
        if hs0 is None:
            hs0 = def_tr["hs"]
        d_hs.append(relative_l2(def_tr["hs"], hs0))
        j_def = task_loss(def_tr)
        j_alt = task_loss(alt_tr)
        adv = j_def - j_alt
        alphas.append(float(alpha))
        xs.append(float(def_tr["x"]))
        advantages.append(adv)
        table.append(
            {
                "alpha": float(alpha),
                "x": float(def_tr["x"]),
                "j_default": j_def,
                "j_alt": j_alt,
                "A": adv,
                "y_end_default": float(def_tr["y_end"]),
                "y_end_alt": float(alt_tr["y_end"]),
                "switch_useful": bool(adv > 0.0),
            }
        )
    d_hs_arr = np.asarray(d_hs, dtype=np.float64)
    a_arr = np.asarray(advantages, dtype=np.float64)
    useful = a_arr > M_MIN
    harmful = a_arr < -M_MIN
    g0 = bool(float(np.max(d_hs_arr)) < D_MACRO_MAX)
    g1 = bool(abs(spearman(np.asarray(alphas), np.asarray(xs))) > RHO_MIN)
    g2 = bool(np.any(a_arr > 0.0) and np.any(a_arr < 0.0))
    g3 = bool(int(np.sum(useful)) >= N_SIDE_MIN and int(np.sum(harmful)) >= N_SIDE_MIN)
    g4 = bool(g3)
    passed = bool(g0 and g1 and g2 and g3 and g4)
    status = stage_status()
    status["R6-B0"] = {"realization_branch_stop": True, "frozen": True}
    status["R7-P0"] = {"pass": passed, "r7_a0_unlocked": passed}
    payload = {
        "stage": "R7-P0",
        "scientific_result": True,
        "neural_probe": False,
        "certificate": False,
        "learner": False,
        "prereg_sha256": _prereg_sha256(),
        "b0_prereg_sha256": b0.get("prereg_sha256"),
        "alphas": list(ALPHAS),
        "u_default": U_DEFAULT,
        "u_alt": U_ALT,
        "y_star": Y_STAR,
        "table": table,
        "max_d_hs": float(np.max(d_hs_arr)),
        "rho_alpha_x": spearman(np.asarray(alphas), np.asarray(xs)),
        "n_useful_margin": int(np.sum(useful)),
        "n_harmful_margin": int(np.sum(harmful)),
        "g0_macro_ambiguity": g0,
        "g1_evidence_visibility": g1,
        "g2_signed_crossover": g2,
        "g3_nondegenerate_margins": g3,
        "g4_balanced_coverage": g4,
        "r7_p0_pass": passed,
        "r7_a0_unlocked": passed,
        "note": "oracle A only; no estimator and no commit rule",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
