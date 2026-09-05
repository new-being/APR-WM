"""R8-P0: active certificate-validity feasibility. No detector, no tau, no pi."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r5_i0 import relative_l2
from .r7_a0 import DELTA, EPS
from .r7_b0 import ALPHAS_TEST, KNOT_IDX, predict_a_wm, predict_knots
from .r7_p0 import BETA_U, D_MACRO_MAX, DT, HOLD_S, PROBE_S, U_ALT, U_DEFAULT, X_SCALE, Y_STAR
from .r7_p1 import F_MAX, _step, simulate_saturated, task_loss
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R8/R8_P0_PREREG.md"
F_MAX_ID = F_MAX
F_MAX_BENIGN = 2.2
F_MAX_INVALID = 1.5
U_EPI = U_DEFAULT
N_EPI = int(KNOT_IDX[0]) + 1
GAP_MIN = 1.0e-3
COST_RATIO_MAX = 1.0
CLASSES = (
    ("id", F_MAX_ID),
    ("benign", F_MAX_BENIGN),
    ("invalid", F_MAX_INVALID),
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_b1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R8-P0 locked until R7-B1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B1":
        raise RuntimeError("R8-P0 locked: B1 summary missing")
    if payload.get("r7_b1_shift_valid"):
        raise RuntimeError("R8-P0 locked: B1 did not invalidate ID calibration")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R8-P0 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R8-P0 locked: frozen B0 WM missing")
    return payload


def s_epi(y_obs: float, y_hat: float) -> float:
    return abs(float(y_obs) - float(y_hat))


def _hold_hs(alpha: float, f_max: float) -> np.ndarray:
    n_hold = int(round(HOLD_S / DT))
    y = 0.0
    v = 0.0
    hs = []
    for _ in range(n_hold):
        y, v, acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
        hs.extend([y, v, acc, 0.0])
    return np.asarray(hs, dtype=np.float64)


def _run_prefix(alpha: float, f_max: float, coefs: np.ndarray) -> dict[str, Any]:
    n_hold = int(round(HOLD_S / DT))
    y = 0.0
    v = 0.0
    for _ in range(n_hold):
        y, v, _acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
    for _ in range(N_EPI):
        y, v, _acc = _step(y, v, U_EPI, alpha, DT, f_max=f_max)
    x = X_SCALE * float(alpha)
    y_hat = float(predict_knots(x, U_EPI, coefs)[0])
    c_prefix = float(y**2 + BETA_U * U_EPI**2)
    y_task = y
    v_task = v
    n_task = int(round(PROBE_S / DT))
    u_task = U_DEFAULT
    for _ in range(n_task):
        y_task, v_task, _acc = _step(y_task, v_task, u_task, alpha, DT, f_max=f_max)
    j_after = float((y_task - Y_STAR) ** 2 + BETA_U * u_task**2)
    j_rest = task_loss(simulate_saturated(alpha, U_DEFAULT, f_max=f_max))
    return {
        "alpha": float(alpha),
        "x": float(x),
        "y_obs": float(y),
        "y_hat": y_hat,
        "S_epi": s_epi(y, y_hat),
        "c_prefix": c_prefix,
        "c_down": float(j_after - j_rest),
        "j_after_default": j_after,
        "j_rest_default": j_rest,
    }


def run_r8_p0(
    output: str | Path,
    *,
    b1_summary: str | Path = "runs/r7_b1/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    b1 = _require_b1(Path(b1_summary))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    x = np.asarray([X_SCALE * a for a in ALPHAS_TEST], dtype=np.float64)
    commit = (predict_a_wm(x, coefs) - q) > DELTA
    commit_alphas = tuple(float(a) for a, flag in zip(ALPHAS_TEST, commit) if flag)
    oracle = {}
    for name, f_max in CLASSES:
        a_true = []
        for alpha in ALPHAS_TEST:
            mid = simulate_saturated(alpha, U_DEFAULT, f_max=f_max)
            alt = simulate_saturated(alpha, U_ALT, f_max=f_max)
            a_true.append(task_loss(mid) - task_loss(alt))
        a_true = np.asarray(a_true, dtype=np.float64)
        ahat = predict_a_wm(x, coefs)
        lower = ahat - q
        n_commit = int(np.sum(commit))
        n_harm = int(np.sum(commit & (a_true <= 0.0)))
        coverage = float(np.mean(a_true >= lower))
        precision = None if n_commit == 0 else float(np.mean(a_true[commit] > 0.0))
        oracle[name] = {
            "f_max": float(f_max),
            "coverage": coverage,
            "precision": precision,
            "n_commit": n_commit,
            "n_harmful_commit": n_harm,
            "n_useful": int(np.sum(a_true >= DELTA)),
            "valid": bool(coverage + 1.0e-12 >= 1.0 - EPS and n_harm == 0),
        }
    hs0 = _hold_hs(float(ALPHAS_TEST[0]), F_MAX_ID)
    d_hs = []
    table = []
    by_class: dict[str, list[float]] = {name: [] for name, _ in CLASSES}
    c_down_id = []
    a_abs_id = []
    for alpha, is_commit in zip(ALPHAS_TEST, commit):
        for name, f_max in CLASSES:
            d_hs.append(relative_l2(_hold_hs(float(alpha), f_max), hs0))
            row = _run_prefix(float(alpha), f_max, coefs)
            row["class"] = name
            row["f_max"] = float(f_max)
            row["commit_set"] = bool(is_commit)
            table.append(row)
            if is_commit:
                by_class[name].append(row["S_epi"])
                if name == "id":
                    c_down_id.append(row["c_down"])
                    mid = simulate_saturated(float(alpha), U_DEFAULT, f_max=F_MAX_ID)
                    alt = simulate_saturated(float(alpha), U_ALT, f_max=F_MAX_ID)
                    a_abs_id.append(abs(task_loss(mid) - task_loss(alt)))
    s_stats = {
        name: {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }
        for name, vals in by_class.items()
    }
    gap = s_stats["invalid"]["min"] - max(s_stats["id"]["max"], s_stats["benign"]["max"])
    mean_c = float(np.mean(c_down_id)) if c_down_id else 0.0
    mean_a = float(np.mean(a_abs_id)) if a_abs_id else 0.0
    cost_ratio = mean_c / mean_a if mean_a > 0 else float("inf")
    g0 = bool(float(np.max(d_hs)) < D_MACRO_MAX)
    g1 = bool(gap + 1.0e-12 >= GAP_MIN)
    g2 = True
    g3 = bool(cost_ratio <= COST_RATIO_MAX + 1.0e-12)
    passed = bool(g0 and g1 and g2 and g3)
    if not g0:
        locus = "passive_indistinguishability"
    elif not g1:
        locus = "validity_alignment"
    elif not g2:
        locus = "timing"
    elif not g3:
        locus = "probe_cost"
    else:
        locus = None
    status = stage_status()
    status["R7-B1"] = {"shift_valid": False, "frozen": True}
    status["R8-P0"] = {"pass": passed, "failure_locus": locus, "r8_a0_unlocked": passed}
    payload = {
        "stage": "R8-P0",
        "scientific_result": True,
        "neural_probe": False,
        "validity_model": False,
        "tau": None,
        "policy": False,
        "wm_updated": False,
        "q_recalibrated": False,
        "u_epi": U_EPI,
        "t_epi": float(N_EPI * DT),
        "n_epi": N_EPI,
        "gap_min": GAP_MIN,
        "cost_ratio_max": COST_RATIO_MAX,
        "f_max_id": F_MAX_ID,
        "f_max_benign": F_MAX_BENIGN,
        "f_max_invalid": F_MAX_INVALID,
        "commit_alphas": list(commit_alphas),
        "prereg_sha256": _prereg_sha256(),
        "b1_prereg_sha256": b1.get("prereg_sha256"),
        "oracle_validity": oracle,
        "s_epi_commit_set": s_stats,
        "gap_commit_set": float(gap),
        "max_d_hs": float(np.max(d_hs)),
        "mean_c_down_id_commit": mean_c,
        "mean_abs_A_id_commit": mean_a,
        "cost_ratio": float(cost_ratio),
        "mean_c_prefix_id_commit": float(
            np.mean([row["c_prefix"] for row in table if row["class"] == "id" and row["commit_set"]])
        ),
        "g0_passive_indistinguishable": g0,
        "g1_validity_aligned_separation": g1,
        "g2_probe_before_commit": g2,
        "g3_probe_cost": g3,
        "r8_p0_pass": passed,
        "r8_a0_unlocked": passed,
        "failure_locus": locus,
        "table": table,
        "note": "S_epi uses only y vs WM; F_max labels only in oracle grouping",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
