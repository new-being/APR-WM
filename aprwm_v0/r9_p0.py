"""R9-P0: amortized calibration feasibility. Frozen P0 probe + R0 PD. No detector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import DELTA
from .r7_b0 import ALPHAS_TEST, predict_a_wm, predict_knots
from .r7_p0 import BETA_U, DT, HOLD_S, PROBE_S, U_ALT, U_DEFAULT, X_SCALE
from .r7_p1 import _step, simulate_saturated, task_loss
from .r8_p0 import CLASSES, F_MAX_ID, GAP_MIN, N_EPI, U_EPI, s_epi
from .r8_p1 import RETURN_MAX
from .r8_r0 import KD, KP, d_state, reset_u
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R9/R9_P0_PREREG.md"
T_TASK = PROBE_S
T_PROBE = float(N_EPI * DT)
T_RECOVER_MAX = 5.0
N_RECOVER_MAX = int(round(T_RECOVER_MAX / DT))
K_MAX = 20


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_r0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R9-P0 locked until R8-R0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R8-R0":
        raise RuntimeError("R9-P0 locked: R0 summary missing")
    if not payload.get("r8_family_stop"):
        raise RuntimeError("R9-P0 locked: R8 same-horizon family is not stopped")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R9-P0 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R9-P0 locked: frozen B0 WM missing")
    return payload


def k_min_from_costs(c_cal: float, v_dec: float) -> int | None:
    if v_dec <= 0.0:
        return None
    k = 1
    while c_cal / float(k) >= v_dec:
        k += 1
        if k > 10**6:
            return None
    return k


def _calibrate(alpha: float, f_max: float, coefs: np.ndarray) -> dict[str, Any]:
    n_hold = int(round(HOLD_S / DT))
    y = 0.0
    v = 0.0
    for _ in range(n_hold):
        y, v, _acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
    u_hist = []
    for _ in range(N_EPI):
        y, v, _acc = _step(y, v, U_EPI, alpha, DT, f_max=f_max)
        u_hist.append(U_EPI)
    y_hat = float(predict_knots(X_SCALE * float(alpha), U_EPI, coefs)[0])
    s = s_epi(float(y), y_hat)
    n_reset = 0
    recovered = d_state(y, v) <= RETURN_MAX + 1.0e-12
    while (not recovered) and n_reset < N_RECOVER_MAX:
        u = reset_u(y, v)
        y, v, _acc = _step(y, v, u, alpha, DT, f_max=f_max)
        u_hist.append(u)
        n_reset += 1
        recovered = d_state(y, v) <= RETURN_MAX + 1.0e-12
    u_arr = np.asarray(u_hist, dtype=np.float64)
    t_recover = float(n_reset * DT)
    t_cal = T_PROBE + t_recover
    return {
        "alpha": float(alpha),
        "S_epi": s,
        "t_recover": t_recover,
        "t_cal": t_cal,
        "recovered": bool(recovered),
        "D_final": d_state(y, v),
        "mean_u2": float(np.mean(u_arr**2)),
    }


def run_r9_p0(
    output: str | Path,
    *,
    r0_summary: str | Path = "runs/r8_r0/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    r0 = _require_r0(Path(r0_summary))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    x = np.asarray([X_SCALE * a for a in ALPHAS_TEST], dtype=np.float64)
    commit = (predict_a_wm(x, coefs) - q) > DELTA
    commit_alphas = tuple(float(a) for a, flag in zip(ALPHAS_TEST, commit) if flag)
    table = []
    by_class_s: dict[str, list[float]] = {name: [] for name, _ in CLASSES}
    by_class_t: dict[str, list[float]] = {name: [] for name, _ in CLASSES}
    a_abs_id = []
    recover_ok = True
    for alpha, is_commit in zip(ALPHAS_TEST, commit):
        if is_commit:
            mid = simulate_saturated(float(alpha), U_DEFAULT, f_max=F_MAX_ID)
            alt = simulate_saturated(float(alpha), U_ALT, f_max=F_MAX_ID)
            a_abs_id.append(abs(task_loss(mid) - task_loss(alt)))
        for name, f_max in CLASSES:
            row = _calibrate(float(alpha), f_max, coefs)
            row["class"] = name
            row["f_max"] = float(f_max)
            row["commit_set"] = bool(is_commit)
            table.append(row)
            if is_commit:
                by_class_s[name].append(row["S_epi"])
                by_class_t[name].append(row["t_recover"])
                if not row["recovered"]:
                    recover_ok = False
    s_stats = {
        name: {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }
        for name, vals in by_class_s.items()
    }
    t_stats = {
        name: {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
        }
        for name, vals in by_class_t.items()
    }
    gap = s_stats["invalid"]["min"] - max(s_stats["id"]["max"], s_stats["benign"]["max"])
    t_recover_max = float(max(t_stats[name]["max"] for name, _ in CLASSES))
    t_cal = T_PROBE + t_recover_max
    v_dec = float(np.mean(a_abs_id)) if a_abs_id else 0.0
    worst = [row for row in table if row["commit_set"] and abs(row["t_recover"] - t_recover_max) < 0.5 * DT]
    mean_u2 = float(np.mean([row["mean_u2"] for row in worst])) if worst else 1.0
    c_time = (t_cal / T_TASK) * v_dec
    c_effort = BETA_U * mean_u2 * (t_cal / T_TASK)
    c_cal = c_time + c_effort
    k_min = k_min_from_costs(c_cal, v_dec)
    g_recover = bool(recover_ok and t_recover_max <= T_RECOVER_MAX + 1.0e-12)
    g_amortize = bool(k_min is not None and k_min <= K_MAX)
    g1 = bool(gap + 1.0e-12 >= GAP_MIN)
    passed = bool(g_recover and g_amortize)
    if not g_recover:
        locus = "recovery_time"
    elif not g_amortize:
        locus = "amortization"
    else:
        locus = None
    if passed:
        pattern = "amortized_feasible"
    elif not g_recover:
        pattern = "recovery_unbounded"
    else:
        pattern = "amortization_too_slow"
    tau_min = None if k_min is None else float(k_min * T_TASK)
    status = stage_status()
    status["R8-R0"] = {"pass": False, "same_horizon_family_stop": True}
    status["R9-P0"] = {
        "pass": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r9_belief_unlocked": passed,
    }
    payload = {
        "stage": "R9-P0",
        "scientific_result": True,
        "neural_probe": False,
        "validity_model": False,
        "tau": None,
        "policy": False,
        "gain_retune": False,
        "persistence": "block_stationary_F_max",
        "u_epi": U_EPI,
        "t_probe": T_PROBE,
        "t_task": T_TASK,
        "t_recover_max_cap": T_RECOVER_MAX,
        "k_max": K_MAX,
        "kp": float(KP),
        "kd": float(KD),
        "return_max": RETURN_MAX,
        "commit_alphas": list(commit_alphas),
        "prereg_sha256": _prereg_sha256(),
        "r0_prereg_sha256": r0.get("prereg_sha256"),
        "s_epi_commit_set": s_stats,
        "gap_commit_set": float(gap),
        "g1_validity_aligned_separation": g1,
        "t_recover_commit_set": t_stats,
        "t_recover_max_commit_all_classes": t_recover_max,
        "t_cal_conservative": t_cal,
        "mean_abs_A_id_commit": v_dec,
        "v_dec": v_dec,
        "c_time": float(c_time),
        "c_effort": float(c_effort),
        "c_cal": float(c_cal),
        "k_min": k_min,
        "tau_validity_min_s": tau_min,
        "g_recover": g_recover,
        "g_amortize": g_amortize,
        "pattern": pattern,
        "r9_p0_pass": passed,
        "r9_belief_unlocked": passed,
        "failure_locus": locus,
        "table": table,
        "note": "oracle amortization; no b(V); R8 same-horizon STOP",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
