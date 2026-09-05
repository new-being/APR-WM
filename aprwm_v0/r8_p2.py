"""R8-P2: damping-aware returning probe. Frozen 3-phase durations. No sweep."""

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
from .r7_p0 import DT, D_MACRO_MAX, U_ALT, U_DEFAULT, X_SCALE
from .r7_p1 import simulate_saturated, task_loss
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
from .r8_p1 import RETURN_MAX, _run_prefix
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R8/R8_P2_PREREG.md"
PHASE_N = (13, 25, 12)
PHASE_SIGN = (1, -1, 1)
DESIGN_D_SEC = (0.02624792, 0.05, 0.02375208)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R8-P2 locked until R8-P1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R8-P1":
        raise RuntimeError("R8-P2 locked: P1 summary missing")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R8-P2 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R8-P2 locked: frozen B0 WM missing")
    return payload


def epi_command() -> np.ndarray:
    assert sum(PHASE_N) == N_EPI
    parts = [np.full(n, sign * U_EPI, dtype=np.float64) for n, sign in zip(PHASE_N, PHASE_SIGN)]
    return np.concatenate(parts)


def run_r8_p2(
    output: str | Path,
    *,
    p1_summary: str | Path = "runs/r8_p1/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    p1 = _require_p1(Path(p1_summary))
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
    if passed:
        pattern = "cheap_validity_evidence"
    elif g_return and g3 and not g1:
        pattern = "returning_but_uninformative"
    elif g1 and (not g_return or not g3):
        pattern = "observable_not_cheap"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R8-P1"] = {"pass": bool(p1.get("r8_p1_pass")), "frozen": True}
    status["R8-P2"] = {
        "pass": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r8_a0_unlocked": passed,
        "open_loop_probe_family_frozen": not passed,
    }
    payload = {
        "stage": "R8-P2",
        "scientific_result": True,
        "neural_probe": False,
        "validity_model": False,
        "tau": None,
        "policy": False,
        "waveform_search": False,
        "selected_by": "max_d_initial_among_damped_return_solutions",
        "u0": U_EPI,
        "t_epi": float(N_EPI * DT),
        "phase_n": list(PHASE_N),
        "phase_sign": list(PHASE_SIGN),
        "design_d_sec": list(DESIGN_D_SEC),
        "return_max": RETURN_MAX,
        "gap_min": GAP_MIN,
        "f_max_id": F_MAX_ID,
        "f_max_benign": F_MAX_BENIGN,
        "f_max_invalid": F_MAX_INVALID,
        "commit_alphas": list(commit_alphas),
        "prereg_sha256": _prereg_sha256(),
        "p1_prereg_sha256": p1.get("prereg_sha256"),
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
        "r8_p2_pass": passed,
        "r8_a0_unlocked": passed,
        "open_loop_probe_family_frozen": not passed,
        "failure_locus": locus,
        "table": table,
        "note": "damped-return 3-phase; d_initial vs 4-phase only; no S_epi tuning",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
