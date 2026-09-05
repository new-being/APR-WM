"""R9-B1-P0: periodic active revalidation. Frozen M=2. No new model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import DELTA
from .r7_b0 import ALPHAS_TEST, predict_a_wm
from .r7_p0 import U_ALT, U_DEFAULT, X_SCALE
from .r7_p1 import simulate_saturated, task_loss
from .r8_p0 import F_MAX_BENIGN, F_MAX_ID, F_MAX_INVALID
from .r9_a0 import K_ORACLE_MIN, TAU_V, _require_b0, c_cal_block, predict_b
from .r9_p0 import _calibrate
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R9/R9_B1_P0_PREREG.md"
M = K_ORACLE_MIN
K_BLOCK = 6
PROBE_K = tuple(range(1, K_BLOCK + 1, M))
K_CHANGE = {"at_probe": 3, "after_probe": 2}
PHASES = ("at_probe", "after_probe")
SEQUENCES = (
    ("stay", F_MAX_ID),
    ("benign", F_MAX_BENIGN),
    ("invalid", F_MAX_INVALID),
)
REVAL_MIN = 0.90


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_r9_b0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R9-B1-P0 locked until R9-B0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R9-B0":
        raise RuntimeError("R9-B1-P0 locked: B0 summary missing")
    return payload


def _require_a0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R9-B1-P0 locked until R9-A0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R9-A0" or not payload.get("r9_a0_go"):
        raise RuntimeError("R9-B1-P0 locked: A0 GO missing")
    return payload


def _f_now(k: int, k_change: int, f_after: float) -> float:
    return F_MAX_ID if k < k_change else float(f_after)


def _run_block(
    *,
    alpha: float,
    f_after: float,
    k_change: int,
    periodic: bool,
    coefs: np.ndarray,
    q: float,
    beta: np.ndarray,
    v_dec: float,
    ell: float,
) -> dict[str, Any]:
    would = bool(ell > DELTA)
    licensed = False
    b = 0.0
    c_sum = 0.0
    n_probe = 0
    k_stale = 0
    n_harm = 0
    n_harm_first = 0
    n_harm_repeat = 0
    gain = 0.0
    probes = []
    tasks = []
    seen_post_commit = False
    for k in range(1, K_BLOCK + 1):
        f_now = _f_now(k, k_change, f_after)
        if periodic and k in PROBE_K:
            cal = _calibrate(alpha, f_now, coefs)
            b = float(predict_b(cal["S_epi"], beta)[0])
            licensed = bool(b >= TAU_V - 1.0e-12)
            c_sum += c_cal_block(cal["t_cal"], cal["mean_u2"], v_dec)
            n_probe += 1
            probes.append({"k": k, "f_max": float(f_now), "b": b, "licensed": licensed, "S_epi": cal["S_epi"]})
        elif (not periodic) and k == 1:
            cal = _calibrate(alpha, F_MAX_ID, coefs)
            b = float(predict_b(cal["S_epi"], beta)[0])
            licensed = bool(b >= TAU_V - 1.0e-12)
            c_sum += c_cal_block(cal["t_cal"], cal["mean_u2"], v_dec)
            n_probe += 1
            probes.append({"k": 1, "f_max": float(F_MAX_ID), "b": b, "licensed": licensed, "S_epi": cal["S_epi"]})
        commit = bool(licensed and would)
        u = U_ALT if commit else U_DEFAULT
        tr = simulate_saturated(alpha, u, f_max=f_now)
        j_pi = task_loss(tr)
        j_def = task_loss(simulate_saturated(alpha, U_DEFAULT, f_max=f_now))
        j_alt = task_loss(simulate_saturated(alpha, U_ALT, f_max=f_now))
        a_now = float(j_def - j_alt)
        post = k >= k_change
        harmful = bool(commit and a_now <= 0.0)
        if commit and post:
            k_stale += 1
        if harmful:
            n_harm += 1
            if post and not seen_post_commit:
                n_harm_first += 1
            elif post and seen_post_commit:
                n_harm_repeat += 1
        if commit and post:
            seen_post_commit = True
        gain += float(j_def - j_pi)
        tasks.append(
            {
                "k": k,
                "f_max": float(f_now),
                "commit": commit,
                "licensed": licensed,
                "harmful": harmful,
                "A": a_now,
            }
        )
    return {
        "alpha": float(alpha),
        "k_stale": int(k_stale),
        "n_probe": n_probe,
        "n_harmful": n_harm,
        "n_harmful_first": n_harm_first,
        "n_harmful_repeat": n_harm_repeat,
        "netvoi": float(gain - c_sum),
        "c_cal_sum": c_sum,
        "probes": probes,
        "tasks": tasks,
    }


def _rate_ok(flags: list[bool], floor: float) -> bool:
    if not flags:
        return False
    return bool(float(np.mean(flags)) + 1.0e-12 >= floor)


def run_r9_b1_p0(
    output: str | Path,
    *,
    b0_r9_summary: str | Path = "runs/r9_b0/formal/summary.json",
    a0_summary: str | Path = "runs/r9_a0/formal/summary.json",
    p0_summary: str | Path = "runs/r9_p0/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    r9b0 = _require_r9_b0(Path(b0_r9_summary))
    a0 = _require_a0(Path(a0_summary))
    p0 = json.loads(Path(p0_summary).read_text(encoding="utf-8"))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    beta = np.asarray(a0["logistic_beta"], dtype=np.float64)
    v_dec = float(p0["v_dec"])
    x = np.asarray([X_SCALE * a for a in ALPHAS_TEST], dtype=np.float64)
    ells = predict_a_wm(x, coefs) - q
    commit_alphas = tuple(float(a) for a, flag in zip(ALPHAS_TEST, ells > DELTA) if flag)
    ell_map = {float(a): float(e) for a, e in zip(ALPHAS_TEST, ells)}
    table = []
    grouped: dict[tuple[str, str, str], list] = {}
    for alpha in commit_alphas:
        ell = ell_map[float(alpha)]
        for seq_name, f_after in SEQUENCES:
            for phase in PHASES:
                k_change = K_CHANGE[phase]
                for policy, periodic in (("periodic", True), ("never", False)):
                    row = _run_block(
                        alpha=float(alpha),
                        f_after=f_after,
                        k_change=k_change,
                        periodic=periodic,
                        coefs=coefs,
                        q=q,
                        beta=beta,
                        v_dec=v_dec,
                        ell=ell,
                    )
                    row.update({"sequence": seq_name, "phase": phase, "policy": policy})
                    table.append(row)
                    grouped.setdefault((seq_name, phase, policy), []).append(row)

    def _probe_after_cut(row: dict[str, Any], k_change: int) -> dict[str, Any] | None:
        for pr in row["probes"]:
            if pr["k"] >= k_change:
                return pr
        return None

    reval_flags = {"stay": [], "benign": [], "invalid": []}
    for seq_name, _f in SEQUENCES:
        for row in grouped[(seq_name, "at_probe", "periodic")]:
            pr = _probe_after_cut(row, K_CHANGE["at_probe"])
            if pr is None:
                reval_flags[seq_name].append(False)
                continue
            if seq_name == "invalid":
                reval_flags[seq_name].append(not pr["licensed"])
            else:
                reval_flags[seq_name].append(bool(pr["licensed"]))
    g_reval = all(_rate_ok(reval_flags[name], REVAL_MIN) for name, _f in SEQUENCES)
    inv_lag = grouped[("invalid", "after_probe", "periodic")]
    max_stale = max(row["k_stale"] for row in inv_lag)
    n_rep = int(sum(row["n_harmful_repeat"] for row in inv_lag))
    g_stale = bool(max_stale <= 1 and n_rep == 0)
    ben_flags = []
    for row in grouped[("benign", "at_probe", "periodic")]:
        pr = _probe_after_cut(row, K_CHANGE["at_probe"])
        ben_flags.append(bool(pr and pr["licensed"]))
    g_benign = _rate_ok(ben_flags, REVAL_MIN)
    stay_net = float(np.mean([row["netvoi"] for row in grouped[("stay", "at_probe", "periodic")]]))
    g_cost = bool(stay_net > 0.0)
    inv_p = float(np.mean([row["netvoi"] for row in inv_lag]))
    inv_n = float(np.mean([row["netvoi"] for row in grouped[("invalid", "after_probe", "never")]]))
    g_shift = bool(inv_p > inv_n)
    passed = bool(g_reval and g_stale and g_benign and g_cost and g_shift)
    if not g_reval:
        locus = "revalidation"
    elif not g_benign:
        locus = "generic_change_detection"
    elif not g_stale:
        locus = "stale_dwell"
    elif not g_cost:
        locus = "surveillance_cost"
    elif not g_shift:
        locus = "shift_value"
    else:
        locus = None
    if passed:
        pattern = "periodic_revalidation_feasible"
    elif g_reval and g_stale and g_benign and g_cost and not g_shift:
        pattern = "dwell_bounded_uneconomic_on_shift"
    elif not g_cost:
        pattern = "safe_but_uneconomic"
    elif not g_stale:
        pattern = "period_too_long"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R9-B0"] = {"go": bool(r9b0.get("r9_b0_go")), "blindness": "policy_induced"}
    status["R9-B1-P0"] = {
        "pass": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r9_b1_unlocked": passed,
    }
    payload = {
        "stage": "R9-B1-P0",
        "scientific_result": True,
        "new_model": False,
        "m_sweep": False,
        "cusum": False,
        "m": M,
        "probe_k": list(PROBE_K),
        "k_change": dict(K_CHANGE),
        "policy_induced_epistemic_blindness": True,
        "commit_alphas": list(commit_alphas),
        "revalidation_rates": {k: float(np.mean(v)) if v else 0.0 for k, v in reval_flags.items()},
        "max_k_stale_invalid_after_probe": int(max_stale),
        "n_harmful_repeat_invalid_after_probe": n_rep,
        "n_harmful_first_invalid_after_probe": int(sum(row["n_harmful_first"] for row in inv_lag)),
        "n_harmful_invalid_after_probe_periodic": int(sum(row["n_harmful"] for row in inv_lag)),
        "n_harmful_invalid_after_probe_never": int(
            sum(row["n_harmful"] for row in grouped[("invalid", "after_probe", "never")])
        ),
        "netvoi_stay_periodic": stay_net,
        "netvoi_invalid_after_probe_periodic": inv_p,
        "netvoi_invalid_after_probe_never": inv_n,
        "g_revalidation": g_reval,
        "g_stale": g_stale,
        "g_benign": g_benign,
        "g_stable_cost": g_cost,
        "g_shift_value": g_shift,
        "pattern": pattern,
        "r9_b1_p0_pass": passed,
        "r9_b1_unlocked": passed,
        "failure_locus": locus,
        "prereg_sha256": _prereg_sha256(),
        "table": table,
        "note": "periodic M=K_min=2; frozen A0 probe; no detector",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
