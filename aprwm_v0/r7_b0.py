"""R7-B0: world-model-mediated certificate. A/J never train the WM."""

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
    RECALL_MIN,
    SHUFFLE_SEED,
    conformal_q,
    _json_float,
)
from .r7_a0 import ALPHAS_CAL as A0_CAL
from .r7_a0 import ALPHAS_FIT as A0_FIT
from .r7_a0 import ALPHAS_TEST as A0_TEST
from .r7_a1 import ALPHAS_CAL as A1_CAL
from .r7_a1 import ALPHAS_FIT as A1_FIT
from .r7_a1 import ALPHAS_TEST as A1_TEST
from .r7_a2 import _eval_predictor, fit_spline, predict_spline
from .r7_p0 import ALPHAS as P0_ALPHAS
from .r7_p0 import BETA_U, U_ALT, U_DEFAULT, X_SCALE, Y_STAR
from .r7_p1 import ALPHAS as P1_ALPHAS
from .r7_p1 import F_MAX, simulate_saturated, task_loss
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_B0_PREREG.md"
KNOT_IDX = (49, 99, 149, 199, 249)
ALPHAS_FIT = (0.42, 0.66, 0.94, 1.14, 1.38, 1.62, 1.86, 2.14, 2.34, 2.58, 2.82, 3.06, 3.34, 3.54)
ALPHAS_CAL = (0.46, 0.74, 0.98, 1.22, 1.42, 1.66, 1.94, 2.18, 2.42, 2.62, 2.86, 3.14, 3.38, 3.62)
ALPHAS_TEST = (
    0.54,
    0.62,
    0.78,
    0.82,
    1.02,
    1.06,
    1.26,
    1.34,
    1.54,
    1.58,
    1.74,
    1.82,
    1.98,
    2.02,
    2.22,
    2.26,
    2.46,
    2.54,
    2.74,
    2.78,
    2.94,
    3.02,
    3.18,
    3.22,
    3.42,
    3.46,
    3.66,
    3.74,
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_a2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-B0 locked until R7-A2 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-A2":
        raise RuntimeError("R7-B0 locked: A2 summary missing")
    if not payload.get("r7_a2_go"):
        raise RuntimeError("R7-B0 locked: A2 GO is false")
    return payload


def _forbidden_alphas() -> set[float]:
    used: set[float] = set()
    for block in (
        P0_ALPHAS,
        P1_ALPHAS,
        A0_FIT,
        A0_CAL,
        A0_TEST,
        A1_FIT,
        A1_CAL,
        A1_TEST,
    ):
        used |= {round(float(a), 4) for a in block}
    return used


def phi(x: float, u: float) -> np.ndarray:
    x = float(x)
    u = float(u)
    return np.asarray([1.0, x, u, x * u, x * x], dtype=np.float64)


def _knots(trace: dict[str, Any]) -> np.ndarray:
    y = np.asarray(trace["y_future"], dtype=np.float64)[:, 0]
    return y[np.asarray(KNOT_IDX)]


def j_from_y_end(y_end: float, u: float) -> float:
    return float((float(y_end) - Y_STAR) ** 2 + BETA_U * float(u) ** 2)


def oracle_row(alpha: float) -> dict[str, Any]:
    mid = simulate_saturated(alpha, U_DEFAULT)
    alt = simulate_saturated(alpha, U_ALT)
    return {
        "alpha": float(alpha),
        "x": float(mid["x"]),
        "A": float(task_loss(mid) - task_loss(alt)),
        "j_default": task_loss(mid),
        "j_alt": task_loss(alt),
        "hs_absmax": float(max(np.max(np.abs(mid["hs"])), np.max(np.abs(alt["hs"])))),
        "y_knots_default": _knots(mid).tolist(),
        "y_knots_alt": _knots(alt).tolist(),
    }


def _pack(alphas: tuple[float, ...], x_perm: np.ndarray | None = None) -> dict[str, Any]:
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


def _wm_samples(pack: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    for i, row in enumerate(pack["rows"]):
        x = float(pack["x"][i])
        for u, key in ((U_DEFAULT, "y_knots_default"), (U_ALT, "y_knots_alt")):
            samples.append(
                {
                    "x": x,
                    "u": float(u),
                    "knots": np.asarray(row[key], dtype=np.float64),
                }
            )
    return samples


def fit_wm(samples: list[dict[str, Any]]) -> np.ndarray:
    design = np.stack([phi(s["x"], s["u"]) for s in samples], axis=0)
    targets = np.stack([s["knots"] for s in samples], axis=0)
    coefs = []
    for k in range(targets.shape[1]):
        coef, *_ = np.linalg.lstsq(design, targets[:, k], rcond=None)
        coefs.append(coef)
    return np.stack(coefs, axis=1)


def predict_knots(x: float, u: float, coefs: np.ndarray) -> np.ndarray:
    return phi(x, u) @ coefs


def predict_a_wm(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    out = []
    for value in np.asarray(x, dtype=np.float64).reshape(-1):
        y_def = predict_knots(float(value), U_DEFAULT, coefs)
        y_alt = predict_knots(float(value), U_ALT, coefs)
        out.append(j_from_y_end(float(y_def[-1]), U_DEFAULT) - j_from_y_end(float(y_alt[-1]), U_ALT))
    return np.asarray(out, dtype=np.float64)


def wm_rmse(samples: list[dict[str, Any]], coefs: np.ndarray) -> float:
    err = []
    for sample in samples:
        pred = predict_knots(sample["x"], sample["u"], coefs)
        err.append(pred - sample["knots"])
    resid = np.concatenate(err)
    return float(np.sqrt(np.mean(resid**2)))


def _eval_wm(
    *,
    fit: dict[str, Any],
    cal: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    samples = _wm_samples(fit)
    coefs = fit_wm(samples)
    ahat_cal = predict_a_wm(cal["x"], coefs)
    q = conformal_q(ahat_cal - cal["A"], EPS)
    ahat = predict_a_wm(test["x"], coefs)
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
        item = {
            "alpha": row["alpha"],
            "x": row["x"],
            "A": row["A"],
            "j_default": row["j_default"],
            "j_alt": row["j_alt"],
            "x_used": float(test["x"][i]),
            "A_hat": _json_float(float(ahat[i])),
            "L": _json_float(float(lower[i])),
            "commit": bool(commit[i]),
            "region": (
                "useful" if row["A"] >= DELTA_USEFUL else ("gray" if row["A"] >= 0.0 else "harmful")
            ),
        }
        rows.append(item)
    return {
        "q": _json_float(q),
        "q_infinite": not np.isfinite(q),
        "coefs": coefs.tolist(),
        "fit_ly_rmse": wm_rmse(samples, coefs),
        "cal_ly_rmse": wm_rmse(_wm_samples(cal), coefs),
        "test_ly_rmse": wm_rmse(_wm_samples(test), coefs),
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


def _pattern(aligned: dict[str, Any]) -> str:
    if not (aligned["g_cert"] and aligned["g_safe"]):
        return "certificate_invalid"
    if aligned["g_recall"] and aligned["g_voi"]:
        return "wm_mediated_positive_certification"
    return "mediation_safe_but_destroys_opportunity"


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


def run_r7_b0(
    output: str | Path,
    *,
    a2_summary: str | Path = "runs/r7_a2/formal/summary.json",
) -> dict[str, Any]:
    a2 = _require_a2(Path(a2_summary))
    forbidden = _forbidden_alphas()
    used = list(ALPHAS_FIT) + list(ALPHAS_CAL) + list(ALPHAS_TEST)
    if any(round(float(a), 4) in forbidden for a in used):
        raise RuntimeError("R7-B0 split collides with A-series or P0/P1 grids")
    if len(set(round(float(a), 4) for a in used)) != len(used):
        raise RuntimeError("R7-B0 split has internal collisions")
    fit = _pack(ALPHAS_FIT)
    cal = _pack(ALPHAS_CAL)
    test = _pack(ALPHAS_TEST)
    hs_absmax = float(max(row["hs_absmax"] for row in fit["rows"] + cal["rows"] + test["rows"]))
    aligned = _eval_wm(fit=fit, cal=cal, test=test)
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = _eval_wm(
        fit=_pack(ALPHAS_FIT, x_perm=rng.permutation(fit["x"])),
        cal=_pack(ALPHAS_CAL, x_perm=rng.permutation(cal["x"])),
        test=_pack(ALPHAS_TEST, x_perm=rng.permutation(test["x"])),
    )
    direct = _eval_predictor(
        fit=fit,
        cal=cal,
        test=test,
        fit_fn=fit_spline,
        predict_fn=predict_spline,
    )
    g_ctrl = bool(aligned["voi"] > shuf["voi"])
    locus = _failure_locus(aligned, g_ctrl)
    go = locus is None
    pattern = _pattern(aligned)
    status = stage_status()
    status["R7-A-series"] = {"frozen": True, "last": "A2"}
    status["R7-B0"] = {"go": go, "pattern": pattern, "failure_locus": locus}
    payload = {
        "stage": "R7-B0",
        "scientific_result": True,
        "neural_probe": False,
        "a_series_frozen": True,
        "certificate_frozen": True,
        "j_in_wm_loss": False,
        "advantage_supervision": False,
        "f_max": F_MAX,
        "delta": DELTA,
        "eps": EPS,
        "knot_idx": list(KNOT_IDX),
        "x_scale": X_SCALE,
        "n_test": len(ALPHAS_TEST),
        "hs_absmax": hs_absmax,
        "prereg_sha256": _prereg_sha256(),
        "a2_prereg_sha256": a2.get("prereg_sha256"),
        "aligned": aligned,
        "shuffled": shuf,
        "direct_diagnostic": {
            "q": direct["q"],
            "coverage": direct["coverage"],
            "precision": direct["precision"],
            "recall_useful": direct["recall_useful"],
            "voi": direct["voi"],
            "n_commit": direct["n_commit"],
            "cal_rmse": direct["cal_rmse"],
        },
        "mediation": {
            "q_wm": aligned["q"],
            "q_direct": direct["q"],
            "recall_wm": aligned["recall_useful"],
            "recall_direct": direct["recall_useful"],
            "a2_historical_q": a2["aligned"]["q"],
            "a2_historical_recall": a2["aligned"]["recall_useful"],
        },
        "g_cert": aligned["g_cert"],
        "g_safe": aligned["g_safe"],
        "g_recall": aligned["g_recall"],
        "g_voi": aligned["g_voi"],
        "g_ctrl": g_ctrl,
        "pattern": pattern,
        "r7_b0_go": go,
        "failure_locus": locus,
        "note": "WM trained on Y knots only; A/J after freeze; A-series not a GO target",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
