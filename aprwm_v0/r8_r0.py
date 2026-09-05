"""R8-R0: billed P0 probe then task-independent PD reset. No detector."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r5_i0 import relative_l2
from .r7_a0 import DELTA, EPS
from .r7_b0 import ALPHAS_TEST, predict_a_wm, predict_knots
from .r7_p0 import BETA_U, DAMPING, D_MACRO_MAX, DT, HOLD_S, PROBE_S, U_ALT, U_DEFAULT, X_SCALE, Y_STAR
from .r7_p1 import _step, simulate_saturated, task_loss
from .r8_p0 import (
    CLASSES,
    COST_RATIO_MAX,
    F_MAX_BENIGN,
    F_MAX_ID,
    F_MAX_INVALID,
    GAP_MIN,
    N_EPI,
    U_EPI,
    _hold_hs,
    s_epi,
)
from .r8_p1 import RETURN_MAX
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R8/R8_R0_PREREG.md"
ALPHA_DES = F_MAX_ID
OMEGA_N = 10.0
ZETA = 1.0
KP = (OMEGA_N**2) / ALPHA_DES
KD = (2.0 * ZETA * OMEGA_N - DAMPING) / ALPHA_DES
U_MAX = 1.0
N_TOTAL = int(round(PROBE_S / DT))
T_TOTAL = float(N_TOTAL * DT)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R8-R0 locked until R8-P2 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R8-P2":
        raise RuntimeError("R8-R0 locked: P2 summary missing")
    if not payload.get("open_loop_probe_family_frozen"):
        raise RuntimeError("R8-R0 locked: open-loop probe family is not frozen")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R8-R0 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R8-R0 locked: frozen B0 WM missing")
    return payload


def reset_u(y: float, v: float) -> float:
    """Nominal PD. State only. Not a validity detector."""
    return float(np.clip(-KP * float(y) - KD * float(v), -U_MAX, U_MAX))


def reset_is_state_only() -> bool:
    names = tuple(inspect.signature(reset_u).parameters)
    return names == ("y", "v")


def d_state(y: float, v: float) -> float:
    return float(np.hypot(y, v))


def _episode_loss(y_end: float, u_seq: np.ndarray) -> float:
    return float((float(y_end) - Y_STAR) ** 2 + BETA_U * float(np.mean(np.asarray(u_seq) ** 2)))


def _run_acquisition(alpha: float, f_max: float, coefs: np.ndarray, u_task: float) -> dict[str, Any]:
    n_hold = int(round(HOLD_S / DT))
    y = 0.0
    v = 0.0
    for _ in range(n_hold):
        y, v, _acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
    u_hist = []
    for _ in range(N_EPI):
        y, v, _acc = _step(y, v, U_EPI, alpha, DT, f_max=f_max)
        u_hist.append(U_EPI)
    x = X_SCALE * float(alpha)
    y_hat = float(predict_knots(x, U_EPI, coefs)[0])
    y_probe = float(y)
    v_probe = float(v)
    s = s_epi(y_probe, y_hat)
    n_left = N_TOTAL - N_EPI
    n_reset = 0
    while n_left > 0 and d_state(y, v) > RETURN_MAX:
        u = reset_u(y, v)
        y, v, _acc = _step(y, v, u, alpha, DT, f_max=f_max)
        u_hist.append(u)
        n_reset += 1
        n_left -= 1
    d_rst = d_state(y, v)
    reset_done = bool(d_rst <= RETURN_MAX + 1.0e-12)
    while n_left > 0:
        y, v, _acc = _step(y, v, u_task, alpha, DT, f_max=f_max)
        u_hist.append(u_task)
        n_left -= 1
    u_arr = np.asarray(u_hist, dtype=np.float64)
    assert len(u_arr) == N_TOTAL
    j_acq = _episode_loss(y, u_arr)
    return {
        "alpha": float(alpha),
        "x": float(x),
        "S_epi": s,
        "y_after_probe": y_probe,
        "v_after_probe": v_probe,
        "D_reset": d_rst,
        "reset_done": reset_done,
        "n_reset": int(n_reset),
        "t_reset": float(n_reset * DT),
        "j_acq": j_acq,
        "y_end": float(y),
        "v_end": float(v),
    }


def run_r8_r0(
    output: str | Path,
    *,
    p2_summary: str | Path = "runs/r8_p2/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    p2 = _require_p2(Path(p2_summary))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    x = np.asarray([X_SCALE * a for a in ALPHAS_TEST], dtype=np.float64)
    commit = (predict_a_wm(x, coefs) - q) > DELTA
    commit_alphas = tuple(float(a) for a, flag in zip(ALPHAS_TEST, commit) if flag)
    g_indep = reset_is_state_only()
    oracle = {}
    for name, f_max in CLASSES:
        a_true = []
        for alpha in ALPHAS_TEST:
            mid = simulate_saturated(alpha, U_DEFAULT, f_max=f_max)
            alt = simulate_saturated(alpha, U_ALT, f_max=f_max)
            a_true.append(task_loss(mid) - task_loss(alt))
        a_true = np.asarray(a_true, dtype=np.float64)
        lower = predict_a_wm(x, coefs) - q
        n_harm = int(np.sum(commit & (a_true <= 0.0)))
        coverage = float(np.mean(a_true >= lower))
        oracle[name] = {
            "f_max": float(f_max),
            "coverage": coverage,
            "n_harmful_commit": n_harm,
            "valid": bool(coverage + 1.0e-12 >= 1.0 - EPS and n_harm == 0),
        }
    hs0 = _hold_hs(float(ALPHAS_TEST[0]), F_MAX_ID)
    d_hs = []
    table = []
    by_class: dict[str, list[float]] = {name: [] for name, _ in CLASSES}
    d_reset_all = []
    t_reset_all = []
    c_epi_id = []
    a_abs_id = []
    for alpha, is_commit in zip(ALPHAS_TEST, commit):
        for name, f_max in CLASSES:
            d_hs.append(relative_l2(_hold_hs(float(alpha), f_max), hs0))
            row = _run_acquisition(float(alpha), f_max, coefs, U_DEFAULT)
            rest = simulate_saturated(float(alpha), U_DEFAULT, f_max=f_max)
            j_rest = task_loss(rest)
            row["class"] = name
            row["f_max"] = float(f_max)
            row["commit_set"] = bool(is_commit)
            row["j_rest_default"] = j_rest
            row["c_epi"] = float(row["j_acq"] - j_rest)
            table.append(row)
            if is_commit:
                by_class[name].append(row["S_epi"])
                d_reset_all.append(row["D_reset"])
                t_reset_all.append(row["t_reset"])
                if name == "id":
                    c_epi_id.append(row["c_epi"])
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
    max_d_reset = float(np.max(d_reset_all)) if d_reset_all else 0.0
    mean_t_reset = float(np.mean(t_reset_all)) if t_reset_all else 0.0
    mean_c = float(np.mean(c_epi_id)) if c_epi_id else 0.0
    mean_a = float(np.mean(a_abs_id)) if a_abs_id else 0.0
    cost_ratio = mean_c / mean_a if mean_a > 0 else float("inf")
    g0 = bool(float(np.max(d_hs)) < D_MACRO_MAX)
    g1 = bool(gap + 1.0e-12 >= GAP_MIN)
    g_reset = bool(max_d_reset <= RETURN_MAX + 1.0e-12)
    g_cost = bool(cost_ratio <= COST_RATIO_MAX + 1.0e-12)
    passed = bool(g1 and g_reset and g_cost and g_indep)
    if not g_indep:
        locus = "reset_uses_validity"
    elif not g1:
        locus = "validity_alignment"
    elif not g_reset:
        locus = "reset_recovery"
    elif not g_cost:
        locus = "acquisition_cost"
    else:
        locus = None
    if passed:
        pattern = "billed_recoverable_evidence"
    elif g1 and g_reset and not g_cost:
        pattern = "recoverable_not_worth"
    elif g1 and not g_reset:
        pattern = "recovery_failure"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R8-P2"] = {
        "pass": bool(p2.get("r8_p2_pass")),
        "open_loop_probe_family_frozen": True,
    }
    status["R8-R0"] = {
        "pass": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r8_a0_unlocked": passed,
        "r8_family_stop": not passed,
    }
    payload = {
        "stage": "R8-R0",
        "scientific_result": True,
        "neural_probe": False,
        "validity_model": False,
        "tau": None,
        "policy": False,
        "wm_updated": False,
        "gain_fit_on_shift": False,
        "u_epi": U_EPI,
        "t_probe": float(N_EPI * DT),
        "n_epi": N_EPI,
        "t_total": T_TOTAL,
        "n_total": N_TOTAL,
        "kp": float(KP),
        "kd": float(KD),
        "omega_n": OMEGA_N,
        "zeta": ZETA,
        "alpha_des": ALPHA_DES,
        "return_max": RETURN_MAX,
        "gap_min": GAP_MIN,
        "cost_ratio_max": COST_RATIO_MAX,
        "f_max_id": F_MAX_ID,
        "f_max_benign": F_MAX_BENIGN,
        "f_max_invalid": F_MAX_INVALID,
        "commit_alphas": list(commit_alphas),
        "prereg_sha256": _prereg_sha256(),
        "p2_prereg_sha256": p2.get("prereg_sha256"),
        "oracle_validity": oracle,
        "s_epi_commit_set": s_stats,
        "gap_commit_set": float(gap),
        "max_d_hs": float(np.max(d_hs)),
        "max_D_reset_commit_all_classes": max_d_reset,
        "mean_t_reset_commit_all_classes": mean_t_reset,
        "mean_c_epi_id_commit": mean_c,
        "mean_abs_A_id_commit": mean_a,
        "cost_ratio": float(cost_ratio),
        "g0_passive_indistinguishable": g0,
        "g1_validity_aligned_separation": g1,
        "g_reset": g_reset,
        "g_cost": g_cost,
        "g_indep_state_only": g_indep,
        "pattern": pattern,
        "r8_r0_pass": passed,
        "r8_a0_unlocked": passed,
        "r8_family_stop": not passed,
        "failure_locus": locus,
        "table": table,
        "note": "P0 probe then nominal PD reset; S_epi before reset; T_total bills reset",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
