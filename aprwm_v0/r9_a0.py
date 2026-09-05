"""R9-A0: persistent certificate-validity belief. Block-stationary. No decay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import DELTA, DELTA_USEFUL, RECALL_MIN, SHUFFLE_SEED, _json_float
from .r7_b0 import ALPHAS_CAL, ALPHAS_FIT, ALPHAS_TEST, predict_a_wm
from .r7_p0 import BETA_U, U_ALT, U_DEFAULT, X_SCALE
from .r7_p1 import simulate_saturated, task_loss
from .r8_p0 import CLASSES, F_MAX_ID
from .r9_p0 import T_TASK, _calibrate
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R9/R9_A0_PREREG.md"
TAU_V = 0.5
RIDGE = 1.0e-8
S_SCALE = 1.0e-3
AUROC_MIN = 0.80
BRIER_MAX = 0.20
K_EVAL = (1, 2, 4, 8)
K_PRIMARY = 2
K_ORACLE_MIN = 2
ALPHAS_VAL_FIT = tuple(float(a) for a in (ALPHAS_FIT + ALPHAS_CAL))


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R9-A0 locked until R9-P0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R9-P0":
        raise RuntimeError("R9-A0 locked: P0 summary missing")
    if not payload.get("r9_p0_pass"):
        raise RuntimeError("R9-A0 locked: P0 did not pass")
    return payload


def _require_b0(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("R9-A0 locked: frozen B0 WM missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0" or not payload.get("r7_b0_go"):
        raise RuntimeError("R9-A0 locked: frozen B0 WM missing")
    return payload


def class_valid(name: str) -> bool:
    return name in ("id", "benign")


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(s: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).reshape(-1) / S_SCALE
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = np.column_stack([np.ones(len(s)), s])
    beta = np.zeros(2, dtype=np.float64)
    eye = np.eye(2)
    for _ in range(80):
        p = sigmoid(x @ beta)
        w = p * (1.0 - p)
        hess = x.T @ (w[:, None] * x) + ridge * eye
        grad = x.T @ (p - y) + ridge * beta
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    return beta


def predict_b(s: np.ndarray | float, beta: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).reshape(-1) / S_SCALE
    x = np.column_stack([np.ones(len(s)), s])
    return sigmoid(x @ beta)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = float(np.sum(pos[:, None] > neg[None, :]))
    eq = float(np.sum(pos[:, None] == neg[None, :]))
    return (gt + 0.5 * eq) / (float(len(pos)) * float(len(neg)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(np.mean((p - y) ** 2))


def c_cal_block(t_cal: float, mean_u2: float, v_dec: float) -> float:
    scale = float(t_cal) / T_TASK
    return float(scale * v_dec + BETA_U * float(mean_u2) * scale)


def _true_a(alpha: float, f_max: float) -> tuple[float, float, float]:
    mid = simulate_saturated(alpha, U_DEFAULT, f_max=f_max)
    alt = simulate_saturated(alpha, U_ALT, f_max=f_max)
    j_def = task_loss(mid)
    j_alt = task_loss(alt)
    return float(j_def - j_alt), float(j_def), float(j_alt)


def _collect(
    alphas: tuple[float, ...],
    coefs: np.ndarray,
    q: float,
    v_dec: float,
) -> list[dict[str, Any]]:
    x = np.asarray([X_SCALE * a for a in alphas], dtype=np.float64)
    l_wm = predict_a_wm(x, coefs) - q
    rows = []
    for alpha, ell in zip(alphas, l_wm):
        would = bool(float(ell) > DELTA)
        for name, f_max in CLASSES:
            cal = _calibrate(float(alpha), f_max, coefs)
            a_true, j_def, j_alt = _true_a(float(alpha), f_max)
            rows.append(
                {
                    "alpha": float(alpha),
                    "class": name,
                    "f_max": float(f_max),
                    "V": int(class_valid(name)),
                    "S_epi": float(cal["S_epi"]),
                    "L": float(ell),
                    "would_commit_b0": would,
                    "A": a_true,
                    "j_default": j_def,
                    "j_alt": j_alt,
                    "t_cal": float(cal["t_cal"]),
                    "t_recover": float(cal["t_recover"]),
                    "recovered": bool(cal["recovered"]),
                    "c_cal": c_cal_block(cal["t_cal"], cal["mean_u2"], v_dec),
                    "useful": bool(a_true >= DELTA_USEFUL),
                }
            )
    return rows


def _attach_belief(rows: list[dict[str, Any]], beta: np.ndarray, s_used: np.ndarray) -> None:
    b = predict_b(s_used, beta)
    for row, value in zip(rows, b):
        row["b"] = float(value)
        row["licensed"] = bool(row["b"] >= TAU_V - 1.0e-12)
        row["commit"] = bool(row["licensed"] and row["would_commit_b0"])


def _netvoi(rows: list[dict[str, Any]], k: int, *, use_c: bool, commit_key: str = "commit") -> float:
    vals = []
    for row in rows:
        gain = float(k) * float(row["A"]) * float(row[commit_key])
        cost = float(row["c_cal"]) if use_c else 0.0
        vals.append(gain - cost)
    return float(np.mean(vals)) if vals else 0.0


def _k_min_realized(rows: list[dict[str, Any]]) -> int | None:
    for k in range(1, 21):
        if _netvoi(rows, k, use_c=True) > 0.0:
            return k
    return None


def run_r9_a0(
    output: str | Path,
    *,
    p0_summary: str | Path = "runs/r9_p0/formal/summary.json",
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    p0 = _require_p0(Path(p0_summary))
    b0 = _require_b0(b0_summary)
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    v_dec = float(p0["v_dec"])
    fit_rows = _collect(ALPHAS_VAL_FIT, coefs, q, v_dec)
    held_rows = _collect(ALPHAS_TEST, coefs, q, v_dec)
    fit_lic = [row for row in fit_rows if row["would_commit_b0"]]
    if len(fit_lic) < 4:
        raise RuntimeError("R9-A0: not enough L>delta fit points for logistic")
    beta = fit_logistic(
        np.asarray([row["S_epi"] for row in fit_lic]),
        np.asarray([row["V"] for row in fit_lic], dtype=np.float64),
    )
    _attach_belief(fit_rows, beta, np.asarray([row["S_epi"] for row in fit_rows]))
    _attach_belief(held_rows, beta, np.asarray([row["S_epi"] for row in held_rows]))
    commit_rows = [row for row in held_rows if row["would_commit_b0"]]
    y_held = np.asarray([row["V"] for row in commit_rows], dtype=np.int32)
    p_held = np.asarray([row["b"] for row in commit_rows], dtype=np.float64)
    auc = auroc(y_held, p_held)
    br = brier(y_held, p_held)
    valid_rows = [row for row in held_rows if row["V"] == 1]
    commits = [row for row in held_rows if row["commit"]]
    n_commit = len(commits)
    if n_commit == 0:
        precision = None
        g_safe = False
    else:
        precision = float(np.mean([row["A"] > 0.0 for row in commits]))
        g_safe = bool(precision >= 1.0 - 1.0e-12)
    useful_valid = [row for row in valid_rows if row["useful"]]
    recall = (
        float(np.mean([row["commit"] for row in useful_valid])) if useful_valid else 0.0
    )
    persist_ok = True
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = [dict(row) for row in commit_rows]
    s_perm = rng.permutation(np.asarray([row["S_epi"] for row in commit_rows]))
    _attach_belief(shuffled, beta, s_perm)
    oracle_commit = []
    for row in commit_rows:
        item = dict(row)
        item["commit"] = bool(row["V"] == 1)
        oracle_commit.append(item)
    b0_commit = []
    for row in commit_rows:
        item = dict(row)
        item["commit"] = True
        b0_commit.append(item)
    netvoi = {
        str(k): {
            "persist": _netvoi(commit_rows, k, use_c=True),
            "oracle": _netvoi(oracle_commit, k, use_c=True),
            "b0": _netvoi(b0_commit, k, use_c=False),
            "shuffle": _netvoi(shuffled, k, use_c=True),
        }
        for k in K_EVAL
    }
    k_real = _k_min_realized(commit_rows)
    delta_k = None if k_real is None else int(k_real - K_ORACLE_MIN)
    g_validity = bool(auc + 1.0e-12 >= AUROC_MIN and br <= BRIER_MAX + 1.0e-12)
    g_use = bool(recall + 1.0e-12 >= RECALL_MIN)
    g_persist = persist_ok
    g_net = bool(netvoi[str(K_PRIMARY)]["persist"] > 0.0)
    passed = bool(g_validity and g_safe and g_use and g_persist and g_net)
    if not g_validity:
        locus = "validity_model"
    elif not g_safe:
        locus = "safety"
    elif not g_use:
        locus = "recall"
    elif not g_persist:
        locus = "persistence"
    elif not g_net:
        locus = "netvoi"
    else:
        locus = None
    if passed:
        pattern = "persistent_license"
    elif g_safe and g_use and g_persist and not g_net:
        pattern = "realized_amortization_gap"
    elif g_validity and not g_safe:
        pattern = "unsafe_license"
    else:
        pattern = "preflight_fail"
    status = stage_status()
    status["R9-P0"] = {"pass": True, "frozen": True, "k_min_oracle": K_ORACLE_MIN}
    status["R9-A0"] = {
        "go": passed,
        "pattern": pattern,
        "failure_locus": locus,
        "r9_b0_unlocked": passed,
    }
    payload = {
        "stage": "R9-A0",
        "scientific_result": True,
        "validity_model": True,
        "decay": False,
        "change_detection": False,
        "reprobe": False,
        "wm_updated": False,
        "q_recalibrated": False,
        "fused_score": False,
        "tau_v": TAU_V,
        "k_eval": list(K_EVAL),
        "k_primary": K_PRIMARY,
        "k_min_oracle": K_ORACLE_MIN,
        "k_min_realized": k_real,
        "delta_k_realization": delta_k,
        "logistic_beta": [float(beta[0]), float(beta[1])],
        "auroc_held": _json_float(auc),
        "brier_held": float(br),
        "precision_commit": precision,
        "n_commit": n_commit,
        "n_commit_set_blocks": len(commit_rows),
        "recall_useful_valid": recall,
        "n_useful_valid": len(useful_valid),
        "netvoi": netvoi,
        "netvoi_k2_persist": netvoi[str(K_PRIMARY)]["persist"],
        "v_dec_p0": v_dec,
        "prereg_sha256": _prereg_sha256(),
        "p0_prereg_sha256": p0.get("prereg_sha256"),
        "g_validity": g_validity,
        "g_safe": g_safe,
        "g_use": g_use,
        "g_persist": g_persist,
        "g_netvoi": g_net,
        "pattern": pattern,
        "r9_a0_go": passed,
        "r9_b0_unlocked": passed,
        "failure_locus": locus,
        "held_rows": held_rows,
        "note": "two-layer b(V) then L; b_{k+1}=b_k; no change detection",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
