"""R7-A2: frozen certificate, one pre-specified spline predictor. No search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import (
    DELTA,
    DELTA_USEFUL,
    EPS,
    SHUFFLE_SEED,
    conformal_q,
    fit_poly,
    predict_a,
    _json_float,
)
from .r7_a1 import ALPHAS_CAL, ALPHAS_FIT, ALPHAS_TEST, _pack
from .r7_p1 import F_MAX
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_A2_PREREG.md"
KNOT_ALPHA = (1.2, 2.0, 2.8)
KNOT_X = tuple(0.25 * float(a) for a in KNOT_ALPHA)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_a1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-A2 locked until R7-A1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-A1":
        raise RuntimeError("R7-A2 locked: A1 summary missing")
    if not payload.get("r7_a1_go"):
        raise RuntimeError("R7-A2 locked: A1 GO is false")
    return payload


def _design_spline(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cols = [np.ones(len(x)), x, x**2, x**3]
    for knot in KNOT_X:
        cols.append(np.maximum(x - knot, 0.0) ** 3)
    return np.stack(cols, axis=1)


def fit_spline(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(_design_spline(x), np.asarray(a, dtype=np.float64), rcond=None)
    return coef


def predict_spline(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return _design_spline(x) @ coef


def _eval_predictor(
    *,
    fit: dict[str, Any],
    cal: dict[str, Any],
    test: dict[str, Any],
    fit_fn,
    predict_fn,
) -> dict[str, Any]:
    coef = fit_fn(fit["x"], fit["A"])
    ahat_cal = predict_fn(cal["x"], coef)
    q = conformal_q(ahat_cal - cal["A"], EPS)
    ahat = predict_fn(test["x"], coef)
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
    cal_rmse = float(np.sqrt(np.mean((ahat_cal - cal["A"]) ** 2)))
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
        "coef": np.asarray(coef, dtype=np.float64).tolist(),
        "cal_rmse": cal_rmse,
        "coverage": coverage,
        "precision": precision,
        "recall_useful": recall,
        "voi": voi,
        "n_commit": n_commit,
        "n_useful": n_useful,
        "vacuous_abstain": vacuous,
        "g_cert": bool(coverage + 1.0e-12 >= 1.0 - EPS),
        "g_safe": g_safe,
        "rows": rows,
    }


def _miss_reason(ahat: float, lower: float) -> str:
    if lower > DELTA:
        return "committed"
    if ahat <= DELTA:
        return "model_bias"
    return "certificate_width"


def _useful_diagnostics(aligned: dict[str, Any]) -> list[dict[str, Any]]:
    q = aligned["q"]
    q_val = float(q) if q is not None else float("inf")
    out = []
    for row in aligned["rows"]:
        if row["region"] != "useful":
            continue
        a_hat = float(row["A_hat"])
        lower = float(row["L"]) if row["L"] is not None else float("-inf")
        a_true = float(row["A"])
        out.append(
            {
                "alpha": row["alpha"],
                "A": a_true,
                "A_hat": a_hat,
                "q": _json_float(q_val),
                "L": row["L"],
                "A_minus_delta": a_true - DELTA,
                "A_hat_minus_delta": a_hat - DELTA,
                "commit": bool(row["commit"]),
                "miss_reason": _miss_reason(a_hat, lower),
            }
        )
    return out


def _pattern(
    *,
    g_cert: bool,
    g_safe: bool,
    recall_up: bool,
    voi_up: bool,
    rmse_down: bool,
    q_down: bool,
) -> str:
    if not (g_cert and g_safe):
        return "certificate_invalid"
    if recall_up and voi_up:
        return "recall_recovery_without_safety_loss"
    if rmse_down or q_down:
        return "better_prediction_not_more_certificates"
    return "no_recall_recovery"


def _failure_locus(
    *,
    g_cert: bool,
    g_safe: bool,
    recall_up: bool,
    voi_up: bool,
    g_ctrl: bool,
) -> str | None:
    if not g_cert:
        return "coverage"
    if not g_safe:
        return "safety"
    if not recall_up:
        return "recall_recovery"
    if not voi_up:
        return "voi_recovery"
    if not g_ctrl:
        return "ctrl"
    return None


def run_r7_a2(
    output: str | Path,
    *,
    a1_summary: str | Path = "runs/r7_a1/formal/summary.json",
) -> dict[str, Any]:
    a1 = _require_a1(Path(a1_summary))
    a1_recall = float(a1["aligned"]["recall_useful"])
    a1_voi = float(a1["aligned"]["voi"])
    a1_q = a1["aligned"]["q"]
    fit = _pack(ALPHAS_FIT)
    cal = _pack(ALPHAS_CAL)
    test = _pack(ALPHAS_TEST)
    baseline = _eval_predictor(
        fit=fit,
        cal=cal,
        test=test,
        fit_fn=fit_poly,
        predict_fn=predict_a,
    )
    aligned = _eval_predictor(
        fit=fit,
        cal=cal,
        test=test,
        fit_fn=fit_spline,
        predict_fn=predict_spline,
    )
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = _eval_predictor(
        fit=_pack(ALPHAS_FIT, x_perm=rng.permutation(fit["x"])),
        cal=_pack(ALPHAS_CAL, x_perm=rng.permutation(cal["x"])),
        test=_pack(ALPHAS_TEST, x_perm=rng.permutation(test["x"])),
        fit_fn=fit_spline,
        predict_fn=predict_spline,
    )
    g_ctrl = bool(aligned["voi"] > shuf["voi"])
    recall_up = bool(aligned["recall_useful"] > a1_recall)
    voi_up = bool(aligned["voi"] > a1_voi)
    rmse_down = bool(aligned["cal_rmse"] < baseline["cal_rmse"])
    q_a2 = aligned["q"]
    q_down = bool(
        q_a2 is not None
        and a1_q is not None
        and float(q_a2) < float(a1_q)
    )
    g_cert = bool(aligned["g_cert"])
    g_safe = bool(aligned["g_safe"])
    locus = _failure_locus(
        g_cert=g_cert,
        g_safe=g_safe,
        recall_up=recall_up,
        voi_up=voi_up,
        g_ctrl=g_ctrl,
    )
    go = locus is None
    pattern = _pattern(
        g_cert=g_cert,
        g_safe=g_safe,
        recall_up=recall_up,
        voi_up=voi_up,
        rmse_down=rmse_down,
        q_down=q_down,
    )
    useful = _useful_diagnostics(aligned)
    n_bias = sum(1 for row in useful if row["miss_reason"] == "model_bias")
    n_width = sum(1 for row in useful if row["miss_reason"] == "certificate_width")
    n_hit = sum(1 for row in useful if row["miss_reason"] == "committed")
    status = stage_status()
    status["R7-A1"] = {"go": True, "frozen": True}
    status["R7-A2"] = {"go": go, "pattern": pattern, "failure_locus": locus}
    payload = {
        "stage": "R7-A2",
        "scientific_result": True,
        "neural_probe": False,
        "certificate_frozen": True,
        "model_search": False,
        "predictor": "cubic_truncated_power_spline",
        "knot_alpha": list(KNOT_ALPHA),
        "knot_x": list(KNOT_X),
        "f_max": F_MAX,
        "delta": DELTA,
        "eps": EPS,
        "n_test": len(ALPHAS_TEST),
        "prereg_sha256": _prereg_sha256(),
        "a1_prereg_sha256": a1.get("prereg_sha256"),
        "a1_recall": a1_recall,
        "a1_voi": a1_voi,
        "a1_q": a1_q,
        "baseline_poly2": {
            "q": baseline["q"],
            "cal_rmse": baseline["cal_rmse"],
            "coverage": baseline["coverage"],
            "precision": baseline["precision"],
            "recall_useful": baseline["recall_useful"],
            "voi": baseline["voi"],
            "n_commit": baseline["n_commit"],
        },
        "aligned": aligned,
        "shuffled": shuf,
        "useful_diagnostics": useful,
        "n_useful_model_bias": n_bias,
        "n_useful_certificate_width": n_width,
        "n_useful_committed": n_hit,
        "rmse_down": rmse_down,
        "q_down": q_down,
        "recall_up": recall_up,
        "voi_up": voi_up,
        "g_cert": g_cert,
        "g_safe": g_safe,
        "g_ctrl": g_ctrl,
        "pattern": pattern,
        "r7_a2_go": go,
        "failure_locus": locus,
        "note": "frozen L=Ahat-q; only f is the cubic spline; A1 split reused",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
