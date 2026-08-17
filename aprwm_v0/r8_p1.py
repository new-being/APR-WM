"""R8-P1: state-neutral four-phase validity probe. No detector, no sweep."""

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
from .r7_b0 import ALPHAS_TEST, predict_a_wm
from .r7_p0 import BETA_U, D_MACRO_MAX, DT, HOLD_S, PROBE_S, U_ALT, U_DEFAULT, X_SCALE, Y_STAR
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
)
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R8/R8_P1_PREREG.md"
PHASE_N = (13, 12, 12, 13)
PHASE_SIGN = (1, -1, -1, 1)
RETURN_MAX = 2.2e-3


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R8-P1 locked until R8-P0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R8-P0":
        raise RuntimeError("R8-P1 locked: P0 summary missing")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R8-P1 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R8-P1 locked: frozen B0 WM missing")
    return payload


def epi_command() -> np.ndarray:
    assert sum(PHASE_N) == N_EPI
    parts = [np.full(n, sign * U_EPI, dtype=np.float64) for n, sign in zip(PHASE_N, PHASE_SIGN)]
    return np.concatenate(parts)


def s_epi(y_obs: np.ndarray, y_hat: np.ndarray) -> float:
    resid = np.asarray(y_obs, dtype=np.float64) - np.asarray(y_hat, dtype=np.float64)
    return float(np.sqrt(np.mean(resid**2)))


def d_terminal(y: float, v: float) -> float:
    return float(np.hypot(y, v))


def _rollout_u(alpha: float, f_max: float, u_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_hold = int(round(HOLD_S / DT))
    y = 0.0
    v = 0.0
    for _ in range(n_hold):
        y, v, _acc = _step(y, v, 0.0, alpha, DT, f_max=f_max)
    ys = []
    vs = []
    for u in u_seq:
        y, v, _acc = _step(y, v, float(u), alpha, DT, f_max=f_max)
        ys.append(y)
        vs.append(v)
    return np.asarray(ys, dtype=np.float64), np.asarray(vs, dtype=np.float64)


def _run_prefix(alpha: float, f_max: float, u_seq: np.ndarray) -> dict[str, Any]:
    y_obs, v_obs = _rollout_u(alpha, f_max, u_seq)
    y_hat, _v_hat = _rollout_u(alpha, F_MAX_ID, u_seq)
    y_t = float(y_obs[-1])
    v_t = float(v_obs[-1])
    y_task = y_t
    v_task = v_t
    n_task = int(round(PROBE_S / DT))
    for _ in range(n_task):
        y_task, v_task, _acc = _step(y_task, v_task, U_DEFAULT, alpha, DT, f_max=f_max)
    j_after = float((y_task - Y_STAR) ** 2 + BETA_U * U_DEFAULT**2)
    j_rest = task_loss(simulate_saturated(alpha, U_DEFAULT, f_max=f_max))
    energy = float(BETA_U * np.mean(u_seq**2))
    return {
        "alpha": float(alpha),
        "x": float(X_SCALE * alpha),
        "S_epi": s_epi(y_obs, y_hat),
        "y_T": y_t,
        "v_T": v_t,
        "D_terminal": d_terminal(y_t, v_t),
        "c_prefix": float(y_t**2 + energy),
        "c_down": float(j_after - j_rest),
        "j_after_default": j_after,
        "j_rest_default": j_rest,
    }


def run_r8_p1(
    output: str | Path,
    *,
    p0_summary: str | Path = "runs/r8_p0/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    p0 = _require_p0(Path(p0_summary))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    u_seq = epi_command()
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
        lower = predict_a_wm(x, coefs) - q
        n_commit = int(np.sum(commit))
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
    d_term_id = []
    c_down_id = []
    a_abs_id = []
    for alpha, is_commit in zip(ALPHAS_TEST, commit):
        for name, f_max in CLASSES:
            d_hs.append(relative_l2(_hold_hs(float(alpha), f_max), hs0))
            row = _run_prefix(float(alpha), f_max, u_seq)
            row["class"] = name
            row["f_max"] = float(f_max)
            row["commit_set"] = bool(is_commit)
            table.append(row)
            if is_commit:
                by_class[name].append(row["S_epi"])
                if name == "id":
                    d_term_id.append(row["D_terminal"])
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
    mean_d = float(np.mean(d_term_id)) if d_term_id else 0.0
    mean_c = float(np.mean(c_down_id)) if c_down_id else 0.0
    mean_a = float(np.mean(a_abs_id)) if a_abs_id else 0.0
    cost_ratio = mean_c / mean_a if mean_a > 0 else float("inf")
    g0 = bool(float(np.max(d_hs)) < D_MACRO_MAX)
    g1 = bool(gap + 1.0e-12 >= GAP_MIN)
    g2 = True
    g_return = bool(mean_d <= RETURN_MAX + 1.0e-12)
    g3 = bool(cost_ratio <= COST_RATIO_MAX + 1.0e-12)
    passed = bool(g0 and g1 and g2 and g_return and g3)
    if not g0:
        locus = "passive_indistinguishability"
    elif not g1:
        locus = "validity_alignment"
    elif not g2:
        locus = "timing"
    elif not g_return:
        locus = "terminal_return"
    elif not g3:
        locus = "probe_cost"
    else:
        locus = None
    if g1 and (not g_return or not g3):
        pattern = "observable_not_cheap"
    elif (g_return and g3) and not g1:
        pattern = "neutrality_observability_conflict"
    elif passed:
        pattern = "cheap_validity_evidence"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R8-P0"] = {"pass": bool(p0.get("r8_p0_pass")), "frozen": True}
    status["R8-P1"] = {"pass": passed, "pattern": pattern, "failure_locus": locus, "r8_a0_unlocked": passed}
    payload = {
        "stage": "R8-P1",
        "scientific_result": True,
        "neural_probe": False,
        "validity_model": False,
        "tau": None,
        "policy": False,
        "waveform_search": False,
        "reset_protocol": False,
        "u0": U_EPI,
        "t_epi": float(N_EPI * DT),
        "phase_n": list(PHASE_N),
        "phase_sign": list(PHASE_SIGN),
        "return_max": RETURN_MAX,
        "gap_min": GAP_MIN,
        "f_max_id": F_MAX_ID,
        "f_max_benign": F_MAX_BENIGN,
        "f_max_invalid": F_MAX_INVALID,
        "commit_alphas": list(commit_alphas),
        "prereg_sha256": _prereg_sha256(),
        "p0_prereg_sha256": p0.get("prereg_sha256"),
        "oracle_validity": oracle,
        "s_epi_commit_set": s_stats,
        "gap_commit_set": float(gap),
        "mean_D_terminal_id_commit": mean_d,
        "max_d_hs": float(np.max(d_hs)),
        "mean_c_down_id_commit": mean_c,
        "mean_abs_A_id_commit": mean_a,
        "cost_ratio": float(cost_ratio),
        "g0_passive_indistinguishable": g0,
        "g1_validity_aligned_separation": g1,
        "g2_probe_before_commit": g2,
        "g_return": g_return,
        "g3_probe_cost": g3,
        "pattern": pattern,
        "r8_p1_pass": passed,
        "r8_a0_unlocked": passed,
        "failure_locus": locus,
        "table": table,
        "note": "only waveform changed; S_epi is RMSE vs ID-plant rollout",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
