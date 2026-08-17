"""R7-A0: one-sided split-conformal switch certificate. No neural probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_p0 import ALPHAS as P0_ALPHAS
from .r7_p0 import U_ALT, U_DEFAULT, simulate_effectiveness, task_loss
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_A0_PREREG.md"
DELTA = 1.0e-3
DELTA_USEFUL = DELTA
EPS = 0.10
RECALL_MIN = 0.25
POLY_DEG = 2
SHUFFLE_SEED = 0
ALPHAS_FIT = (0.55, 0.85, 1.15, 1.45, 1.75, 2.05, 2.35, 2.65, 2.95, 3.25)
ALPHAS_CAL = (0.65, 0.95, 1.25, 1.55, 1.85, 2.15, 2.45, 2.75, 3.05, 3.35)
ALPHAS_TEST = (0.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-A0 locked until R7-P0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-P0":
        raise RuntimeError("R7-A0 locked: P0 summary missing")
    if not payload.get("r7_p0_pass"):
        raise RuntimeError("R7-A0 locked: P0 did not pass")
    return payload


def oracle_row(alpha: float) -> dict[str, Any]:
    mid = simulate_effectiveness(alpha, U_DEFAULT)
    cons = simulate_effectiveness(alpha, U_ALT)
    adv = task_loss(mid) - task_loss(cons)
    return {
        "alpha": float(alpha),
        "x": float(mid["x"]),
        "A": float(adv),
        "j_default": task_loss(mid),
        "j_alt": task_loss(cons),
    }


def _design(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cols = [np.ones(len(x))]
    for power in range(1, POLY_DEG + 1):
        cols.append(x**power)
    return np.stack(cols, axis=1)


def fit_poly(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(_design(x), np.asarray(a, dtype=np.float64), rcond=None)
    return coef


def predict_a(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return _design(x) @ coef


def conformal_q(scores: np.ndarray, eps: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - eps)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def _json_float(value: float) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _pack(alphas: tuple[float, ...], x_perm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rows = [oracle_row(alpha) for alpha in alphas]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    if x_perm is not None:
        x = np.asarray(x_perm, dtype=np.float64)
    return {
        "alpha": np.asarray([row["alpha"] for row in rows], dtype=np.float64),
        "x": x,
        "A": np.asarray([row["A"] for row in rows], dtype=np.float64),
        "rows": rows,
    }


def _eval_split(
    *,
    fit: dict[str, np.ndarray],
    cal: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> dict[str, Any]:
    coef = fit_poly(fit["x"], fit["A"])
    ahat_cal = predict_a(cal["x"], coef)
    q = conformal_q(ahat_cal - cal["A"], EPS)
    ahat = predict_a(test["x"], coef)
    lower = ahat - q
    commit = lower > DELTA
    a_true = test["A"]
    useful = a_true >= DELTA_USEFUL
    n_commit = int(np.sum(commit))
    n_useful = int(np.sum(useful))
    coverage = float(np.mean(a_true >= lower)) if len(a_true) else 0.0
    if n_commit == 0:
        precision = None
        g_safe = False
        vacuous = True
    else:
        precision = float(np.mean(a_true[commit] > 0.0))
        g_safe = bool(precision >= 1.0 - 1.0e-12)
        vacuous = False
    recall = float(np.mean(commit[useful])) if n_useful else 0.0
    voi = float(np.mean(a_true * commit))
    rows = []
    for i, row in enumerate(test["rows"]):
        item = dict(row)
        item["x_used"] = float(test["x"][i])
        item["A_hat"] = _json_float(float(ahat[i]))
        item["L"] = _json_float(float(lower[i]))
        item["commit"] = bool(commit[i])
        item["region"] = (
            "useful" if row["A"] >= DELTA_USEFUL else ("gray" if row["A"] >= 0.0 else "harmful")
        )
        rows.append(item)
    return {
        "q": _json_float(q),
        "q_infinite": not np.isfinite(q),
        "coef": coef.tolist(),
        "coverage": coverage,
        "precision": precision,
        "recall_useful": recall,
        "voi": voi,
        "n_commit": n_commit,
        "n_useful": n_useful,
        "vacuous_abstain": vacuous,
        "g_cert": bool(coverage + 1.0e-12 >= 1.0 - EPS),
        "g_safe": g_safe,
        "g_recall": bool(recall + 1.0e-12 >= RECALL_MIN),
        "g_voi": bool(voi > 0.0),
        "rows": rows,
    }


def _failure_locus(aligned: dict[str, Any], g_ctrl: bool) -> str | None:
    if not aligned["g_cert"]:
        return "coverage"
    if not aligned["g_safe"]:
        return "safety"
    if not aligned["g_recall"]:
        return "recall"
    if not aligned["g_voi"]:
        return "voi"
    if not g_ctrl:
        return "ctrl"
    return None


def run_r7_a0(
    output: str | Path,
    *,
    p0_summary: str | Path = "runs/r7_p0/formal/summary.json",
) -> dict[str, Any]:
    p0 = _require_p0(Path(p0_summary))
    p0_set = {round(float(x), 4) for x in P0_ALPHAS}
    for split in (ALPHAS_FIT, ALPHAS_CAL, ALPHAS_TEST):
        if any(round(float(a), 4) in p0_set for a in split):
            raise RuntimeError("R7-A0 split collides with P0 grid")
    fit = _pack(ALPHAS_FIT)
    cal = _pack(ALPHAS_CAL)
    test = _pack(ALPHAS_TEST)
    aligned = _eval_split(fit=fit, cal=cal, test=test)
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = _eval_split(
        fit=_pack(ALPHAS_FIT, x_perm=rng.permutation(fit["x"])),
        cal=_pack(ALPHAS_CAL, x_perm=rng.permutation(cal["x"])),
        test=_pack(ALPHAS_TEST, x_perm=rng.permutation(test["x"])),
    )
    g_ctrl = bool(aligned["voi"] > shuf["voi"])
    locus = _failure_locus(aligned, g_ctrl)
    go = locus is None
    status = stage_status()
    status["R7-P0"] = {"pass": True, "frozen": True}
    status["R7-A0"] = {"go": go, "failure_locus": locus}
    payload = {
        "stage": "R7-A0",
        "scientific_result": True,
        "neural_probe": False,
        "symmetric_z": False,
        "delta": DELTA,
        "eps": EPS,
        "prereg_sha256": _prereg_sha256(),
        "p0_prereg_sha256": p0.get("prereg_sha256"),
        "aligned": aligned,
        "shuffled": shuf,
        "g_cert": aligned["g_cert"],
        "g_safe": aligned["g_safe"],
        "g_recall": aligned["g_recall"],
        "g_voi": aligned["g_voi"],
        "g_ctrl": g_ctrl,
        "r7_a0_go": go,
        "failure_locus": locus,
        "note": "one-sided split-conformal L=Ahat-q; P0 grid unused",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
