"""R9-B0: passive unidirectional revocation of a stale validity license."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import DELTA
from .r7_b0 import (
    ALPHAS_CAL,
    ALPHAS_FIT,
    ALPHAS_TEST,
    KNOT_IDX,
    _knots,
    predict_a_wm,
    predict_knots,
)
from .r7_p0 import U_ALT, U_DEFAULT, X_SCALE
from .r7_p1 import simulate_saturated, task_loss
from .r8_p0 import F_MAX_BENIGN, F_MAX_ID, F_MAX_INVALID
from .r9_a0 import TAU_V, _require_b0, c_cal_block, predict_b
from .r9_p0 import _calibrate
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R9/R9_B0_PREREG.md"
K_CUT = 2
K_POST = 4
K_BLOCK = K_CUT + K_POST
CUSUM_K = 0.5
CUSUM_H = 4.0
SIGMA_FLOOR = 1.0e-8
STABLE_MIN = 0.90
REVOKE_M = 1
REVOKE_MIN = 0.80
SEQUENCES = (
    ("stay", F_MAX_ID),
    ("benign", F_MAX_BENIGN),
    ("invalid", F_MAX_INVALID),
)
ALPHAS_DEV = tuple(float(a) for a in (ALPHAS_FIT + ALPHAS_CAL))


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_a0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R9-B0 locked until R9-A0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R9-A0":
        raise RuntimeError("R9-B0 locked: A0 summary missing")
    if not payload.get("r9_a0_go"):
        raise RuntimeError("R9-B0 locked: A0 GO is false")
    return payload


def residual(alpha: float, u: float, f_max: float, coefs: np.ndarray) -> tuple[float, dict[str, Any]]:
    trace = simulate_saturated(alpha, u, f_max=f_max)
    y_obs = _knots(trace)
    y_hat = predict_knots(X_SCALE * float(alpha), float(u), coefs)
    r = float(np.sqrt(np.mean((y_obs - y_hat) ** 2)))
    return r, trace


def cusum_step(e: float, r: float, mu: float, sigma: float) -> float:
    s = (float(r) - mu) / sigma - CUSUM_K
    return float(max(0.0, e + s))


def _dev_mu_sigma(coefs: np.ndarray, q: float) -> tuple[float, float, list[float]]:
    x = np.asarray([X_SCALE * a for a in ALPHAS_DEV], dtype=np.float64)
    ell = predict_a_wm(x, coefs) - q
    rs = []
    for alpha, flag in zip(ALPHAS_DEV, ell > DELTA):
        if not flag:
            continue
        r, _tr = residual(float(alpha), U_ALT, F_MAX_ID, coefs)
        rs.append(r)
    rs_arr = np.asarray(rs, dtype=np.float64)
    mu = float(np.mean(rs_arr))
    sigma = float(max(float(np.std(rs_arr, ddof=1)) if len(rs_arr) > 1 else 0.0, SIGMA_FLOOR))
    return mu, sigma, rs


def _run_block(
    *,
    alpha: float,
    f_after: float,
    coefs: np.ndarray,
    q: float,
    mu: float,
    sigma: float,
    cal: dict[str, Any],
    b0: float,
    ell: float,
    revoke: bool,
) -> dict[str, Any]:
    licensed = bool(b0 >= TAU_V - 1.0e-12)
    would = bool(ell > DELTA)
    e = 0.0
    k_revoke = None
    tasks = []
    n_harm = 0
    n_harm_first = 0
    n_harm_repeat = 0
    n_stale = 0
    gain = 0.0
    for k in range(1, K_BLOCK + 1):
        f_now = F_MAX_ID if k <= K_CUT else f_after
        post_j = None if k <= K_CUT else k - K_CUT
        commit = bool(licensed and would)
        u = U_ALT if commit else U_DEFAULT
        r, trace = residual(alpha, u, f_now, coefs)
        j_pi = task_loss(trace)
        j_def = task_loss(simulate_saturated(alpha, U_DEFAULT, f_max=f_now))
        j_alt = task_loss(simulate_saturated(alpha, U_ALT, f_max=f_now))
        a_now = float(j_def - j_alt)
        harmful = bool(commit and a_now <= 0.0)
        if harmful:
            n_harm += 1
            if post_j == 1:
                n_harm_first += 1
            elif post_j is not None and post_j >= 2:
                n_harm_repeat += 1
        if commit and post_j is not None:
            n_stale += 1
        gain += float(j_def - j_pi)
        tasks.append(
            {
                "k": k,
                "f_max": float(f_now),
                "commit": commit,
                "licensed": licensed,
                "u": float(u),
                "r": r,
                "E": e,
                "A": a_now,
                "harmful": harmful,
                "j_pi": j_pi,
                "j_default": j_def,
            }
        )
        if licensed and revoke:
            e = cusum_step(e, r, mu, sigma)
            tasks[-1]["E"] = e
            if e >= CUSUM_H - 1.0e-12:
                licensed = False
                if post_j is not None and k_revoke is None:
                    k_revoke = int(post_j)
                elif post_j is None and k_revoke is None:
                    k_revoke = 0
    c_cal = float(cal["c_cal"])
    return {
        "alpha": float(alpha),
        "b0": b0,
        "c_cal": c_cal,
        "k_revoke": k_revoke,
        "retained": bool(licensed),
        "n_harmful": n_harm,
        "n_harmful_first": n_harm_first,
        "n_harmful_repeat": n_harm_repeat,
        "n_stale_commits": n_stale,
        "netvoi": float(gain - c_cal),
        "tasks": tasks,
    }


def _seq_stats(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(blocks)
    retain = float(np.mean([row["retained"] for row in blocks])) if n else 0.0
    kr = [row["k_revoke"] for row in blocks if row["k_revoke"] is not None]
    on_time = float(np.mean([row["k_revoke"] is not None and 1 <= row["k_revoke"] <= REVOKE_M for row in blocks])) if n else 0.0
    return {
        "n": n,
        "p_retain": retain,
        "p_revoke_within_m": on_time,
        "mean_k_revoke": None if not kr else float(np.mean(kr)),
        "n_harmful": int(sum(row["n_harmful"] for row in blocks)),
        "n_harmful_first": int(sum(row["n_harmful_first"] for row in blocks)),
        "n_harmful_repeat": int(sum(row["n_harmful_repeat"] for row in blocks)),
        "n_stale_commits": int(sum(row["n_stale_commits"] for row in blocks)),
        "mean_netvoi": float(np.mean([row["netvoi"] for row in blocks])) if n else 0.0,
    }


def run_r9_b0(
    output: str | Path,
    *,
    a0_summary: str | Path = "runs/r9_a0/formal/summary.json",
    p0_summary: str | Path = "runs/r9_p0/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    a0 = _require_a0(Path(a0_summary))
    if not Path(p0_summary).is_file():
        raise RuntimeError("R9-B0 locked: R9-P0 summary missing")
    p0 = json.loads(Path(p0_summary).read_text(encoding="utf-8"))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    beta = np.asarray(a0["logistic_beta"], dtype=np.float64)
    v_dec = float(p0["v_dec"])
    mu, sigma, dev_r = _dev_mu_sigma(coefs, q)
    x = np.asarray([X_SCALE * a for a in ALPHAS_TEST], dtype=np.float64)
    commit_alphas = tuple(float(a) for a, flag in zip(ALPHAS_TEST, (predict_a_wm(x, coefs) - q) > DELTA) if flag)
    cals = {}
    for alpha in commit_alphas:
        cal = _calibrate(float(alpha), F_MAX_ID, coefs)
        b_init = float(predict_b(cal["S_epi"], beta)[0])
        ell = float(predict_a_wm(np.asarray([X_SCALE * alpha]), coefs)[0] - q)
        cal["c_cal"] = c_cal_block(cal["t_cal"], cal["mean_u2"], v_dec)
        cals[float(alpha)] = (cal, b_init, ell)
    table = []
    grouped: dict[str, dict[str, list]] = {
        name: {"revoke": [], "persist": []} for name, _f in SEQUENCES
    }
    for alpha in commit_alphas:
        for seq_name, f_after in SEQUENCES:
            cal, b_init, ell = cals[float(alpha)]
            rev = _run_block(
                alpha=alpha,
                f_after=f_after,
                coefs=coefs,
                q=q,
                mu=mu,
                sigma=sigma,
                cal=cal,
                b0=b_init,
                ell=ell,
                revoke=True,
            )
            per = _run_block(
                alpha=alpha,
                f_after=f_after,
                coefs=coefs,
                q=q,
                mu=mu,
                sigma=sigma,
                cal=cal,
                b0=b_init,
                ell=ell,
                revoke=False,
            )
            rev["sequence"] = seq_name
            per["sequence"] = seq_name
            rev["policy"] = "revoke"
            per["policy"] = "persist"
            grouped[seq_name]["revoke"].append(rev)
            grouped[seq_name]["persist"].append(per)
            table.append(rev)
            table.append(per)
    by_seq = {
        name: {
            "revoke": _seq_stats(grouped[name]["revoke"]),
            "persist": _seq_stats(grouped[name]["persist"]),
        }
        for name, _f in SEQUENCES
    }
    all_rev = [row for name, _f in SEQUENCES for row in grouped[name]["revoke"]]
    all_per = [row for name, _f in SEQUENCES for row in grouped[name]["persist"]]
    net_rev = float(np.mean([row["netvoi"] for row in all_rev]))
    net_per = float(np.mean([row["netvoi"] for row in all_per]))
    n_h_rev = by_seq["invalid"]["revoke"]["n_harmful"]
    n_h_per = by_seq["invalid"]["persist"]["n_harmful"]
    g_stable = bool(by_seq["stay"]["revoke"]["p_retain"] + 1.0e-12 >= STABLE_MIN)
    g_benign = bool(by_seq["benign"]["revoke"]["p_retain"] + 1.0e-12 >= STABLE_MIN)
    g_revoke = bool(by_seq["invalid"]["revoke"]["p_revoke_within_m"] + 1.0e-12 >= REVOKE_MIN)
    g_harm = bool(n_h_rev < n_h_per)
    g_net = bool(net_rev > net_per)
    passed = bool(g_stable and g_benign and g_revoke and g_harm and g_net)
    if not g_stable:
        locus = "false_revoke_id"
    elif not g_benign:
        locus = "generic_change_detection"
    elif not g_revoke:
        if by_seq["invalid"]["revoke"]["p_retain"] >= 1.0 - 1.0e-12:
            locus = "ordinary_task_insufficient"
        else:
            locus = "revoke_too_late"
    elif not g_harm:
        locus = "harm_not_reduced"
    elif not g_net:
        locus = "netvoi"
    else:
        locus = None
    if passed:
        pattern = "passive_validity_revocation"
    elif not g_benign and g_revoke:
        pattern = "generic_change_detection"
    elif not g_revoke:
        pattern = "passive_evidence_missing_or_late"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R9-A0"] = {"go": True, "frozen": True}
    status["R9-B0"] = {
        "go": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r9_b1_unlocked": passed,
    }
    payload = {
        "stage": "R9-B0",
        "scientific_result": True,
        "reprobe": False,
        "decay": False,
        "bidirectional_belief": False,
        "score_uses_A": False,
        "k_cut": K_CUT,
        "k_block": K_BLOCK,
        "cusum_k": CUSUM_K,
        "cusum_h": CUSUM_H,
        "mu_dev_id": mu,
        "sigma_dev_id": sigma,
        "n_dev_id": len(dev_r),
        "tau_v": TAU_V,
        "commit_alphas": list(commit_alphas),
        "by_sequence": by_seq,
        "netvoi_revoke": net_rev,
        "netvoi_persist": net_per,
        "n_harmful_revoke_invalid": n_h_rev,
        "n_harmful_persist_invalid": n_h_per,
        "n_harmful_repeat_revoke_invalid": by_seq["invalid"]["revoke"]["n_harmful_repeat"],
        "g_stable": g_stable,
        "g_benign": g_benign,
        "g_revoke": g_revoke,
        "g_harm": g_harm,
        "g_netvoi": g_net,
        "pattern": pattern,
        "r9_b0_go": passed,
        "r9_b1_unlocked": passed,
        "failure_locus": locus,
        "prereg_sha256": _prereg_sha256(),
        "a0_prereg_sha256": a0.get("prereg_sha256"),
        "table": table,
        "note": "passive CUSUM revoke; no re-probe; first post-shift exposure allowed",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
