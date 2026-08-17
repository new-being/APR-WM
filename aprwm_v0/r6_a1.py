"""R6-A1: nested-LOO pairwise margin-uncertainty audit. No planner change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r5_d0 import _choose, _fit_action_models, _oracle_bank, _predict_y
from .r5_d0_preflight import ACTIONS, task_loss_y
from .r5_i0_selfstress import LAMBDAS
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R6/R6_A1_PREREG.md"
Z_LOW = 1.0
SIGMA_FLOOR = 1.0e-18
NAMES = tuple(name for name, _ in ACTIONS)
PAIRS = tuple((a, b) for a in NAMES for b in NAMES if a != b)
KEY_LAMBDAS = (0.0, 2.0, 12.0)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_a0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R6-A1 locked until R6-A0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R6-A0":
        raise RuntimeError("R6-A1 locked: A0 summary missing")
    return payload


def _require_d0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R6-A1 locked until R5-D0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R5-D0":
        raise RuntimeError("R6-A1 locked: D0 summary missing")
    return payload


def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def _rms(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr**2)))


def _mad(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(arr - np.median(arr))))


def _j_of(pred_y: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: task_loss_y(pred_y[name]) for name in NAMES}


def _margins(js: dict[str, float]) -> dict[str, float]:
    return {pair_key(a, b): float(js[b] - js[a]) for a, b in PAIRS}


def _inner_scale(
    bank: dict[float, dict[str, Any]],
    train: list[float],
    *,
    x_of: dict[float, float],
    with_x: bool,
    y_shape: tuple[int, int],
) -> dict[str, Any]:
    etas: dict[str, list[float]] = {pair_key(a, b): [] for a, b in PAIRS}
    for held in train:
        inner_train = [lam for lam in train if lam != held]
        models = _fit_action_models(bank, inner_train, x_of=x_of, with_x=with_x)
        x = None if not with_x else x_of[held]
        pred = _predict_y(models, bank[held]["hs"], x, y_shape)
        hat = _margins(_j_of(pred))
        tru = _margins({name: float(bank[held]["j"][name]) for name in NAMES})
        for key in etas:
            etas[key].append(hat[key] - tru[key])
    sigma = {key: max(_rms(vals), SIGMA_FLOOR) for key, vals in etas.items()}
    mad = {key: _mad(vals) for key, vals in etas.items()}
    return {"sigma": sigma, "mad": mad, "eta": etas}


def _pair_rows(
    *,
    hat_m: dict[str, float],
    true_m: dict[str, float],
    sigma: dict[str, float],
    mad: dict[str, float],
) -> dict[str, dict[str, Any]]:
    rows = {}
    for a, b in PAIRS:
        key = pair_key(a, b)
        hat = hat_m[key]
        sig = sigma[key]
        z = abs(hat) / sig
        rows[key] = {
            "a": a,
            "b": b,
            "m": true_m[key],
            "hat_m": hat,
            "sigma": sig,
            "mad": mad[key],
            "z": z,
            "crosses_zero": bool(z < Z_LOW),
            "sign_error": bool(true_m[key] * hat < 0.0),
        }
    return rows


def _audit_planner(
    *,
    bank: dict[float, dict[str, Any]],
    held: float,
    pred_y: dict[str, np.ndarray],
    scale: dict[str, Any],
    d0_hat: str,
) -> dict[str, Any]:
    star = bank[held]["a_star"]
    j_hat = _j_of(pred_y)
    hat_a = _choose(j_hat)
    hat_m = _margins(j_hat)
    true_m = _margins({name: float(bank[held]["j"][name]) for name in NAMES})
    pairs = _pair_rows(hat_m=hat_m, true_m=true_m, sigma=scale["sigma"], mad=scale["mad"])
    preserved = hat_a == star
    decide_key = None if preserved or hat_a is None else pair_key(star, hat_a)
    decide = None if decide_key is None else pairs[decide_key]
    return {
        "lam": held,
        "a_star": star,
        "hat_a": hat_a,
        "preserved": preserved,
        "d0_match": hat_a == d0_hat,
        "deciding_pair": decide_key,
        "z_deciding": None if decide is None else decide["z"],
        "lowconf_deciding": None if decide is None else decide["crosses_zero"],
        "confident_wrong": bool(not preserved and decide is not None and decide["z"] >= Z_LOW),
        "pairs": pairs,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _planner_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if not row["preserved"]]
    preserved = [row for row in rows if row["preserved"]]
    z_err = [float(row["z_deciding"]) for row in errors if row["z_deciding"] is not None]
    z_ok = []
    for row in preserved:
        star = row["a_star"]
        for a, b in PAIRS:
            if a != star:
                continue
            z_ok.append(float(row["pairs"][pair_key(a, b)]["z"]))
    n_err = len(errors)
    n_low = int(sum(bool(row["lowconf_deciding"]) for row in errors))
    n_conf = int(sum(bool(row["confident_wrong"]) for row in errors))
    u1 = None if n_err == 0 else n_low / n_err
    med_err = _median(z_err)
    med_ok = _median(z_ok)
    u2 = None if med_err is None or med_ok is None else bool(med_err < med_ok)
    return {
        "n_rank_err": n_err,
        "n_lowconf_err": n_low,
        "n_confident_wrong": n_conf,
        "u1_lowconf_fraction": u1,
        "u2_errors_less_confident": u2,
        "median_z_error": med_err,
        "median_z_preserved_star_pairs": med_ok,
    }


def run_r6_a1(
    output: str | Path,
    *,
    a0_summary: str | Path = "runs/r6_a0/formal/summary.json",
    d0_summary: str | Path = "runs/r5_d0/formal/summary.json",
) -> dict[str, Any]:
    a0 = _require_a0(Path(a0_summary))
    d0 = _require_d0(Path(d0_summary))
    bank = _oracle_bank()
    lams = [float(x) for x in LAMBDAS]
    y_shape = next(iter(bank.values()))["y"]["mid"].shape
    x_true = {lam: bank[lam]["x"] for lam in lams}
    tables = {"p0": [], "px": []}
    for held in lams:
        train = [lam for lam in lams if lam != held]
        scale0 = _inner_scale(bank, train, x_of=x_true, with_x=False, y_shape=y_shape)
        scalex = _inner_scale(bank, train, x_of=x_true, with_x=True, y_shape=y_shape)
        m0 = _fit_action_models(bank, train, x_of=x_true, with_x=False)
        mx = _fit_action_models(bank, train, x_of=x_true, with_x=True)
        hs = bank[held]["hs"]
        y0 = _predict_y(m0, hs, None, y_shape)
        yx = _predict_y(mx, hs, x_true[held], y_shape)
        d0_row = next(item for item in d0["rows"] if float(item["lam"]) == held)
        tables["p0"].append(
            _audit_planner(bank=bank, held=held, pred_y=y0, scale=scale0, d0_hat=d0_row["a0"])
        )
        tables["px"].append(
            _audit_planner(bank=bank, held=held, pred_y=yx, scale=scalex, d0_hat=d0_row["ax"])
        )

    stats0 = _planner_stats(tables["p0"])
    statsx = _planner_stats(tables["px"])
    d0_match = bool(all(row["d0_match"] for row in tables["p0"] + tables["px"]))
    px_err = [row for row in tables["px"] if not row["preserved"]]
    px_all_low = bool(px_err) and all(bool(row["lowconf_deciding"]) for row in px_err)
    lam2 = next(row for row in tables["px"] if row["lam"] == 2.0)
    lam2_confident_wrong = bool(lam2["confident_wrong"])
    if lam2_confident_wrong or (px_err and not px_all_low):
        fork = "allocation_or_mixed"
        b0_licensed = False
    elif px_all_low:
        fork = "b0_candidate"
        b0_licensed = True
    else:
        fork = "no_px_error"
        b0_licensed = False

    cases = {}
    for lam in KEY_LAMBDAS:
        row = next(item for item in tables["px"] if item["lam"] == lam)
        cases[str(int(lam))] = {
            "a_star": row["a_star"],
            "hat_a": row["hat_a"],
            "preserved": row["preserved"],
            "deciding_pair": row["deciding_pair"],
            "z_deciding": row["z_deciding"],
            "lowconf_deciding": row["lowconf_deciding"],
            "confident_wrong": row["confident_wrong"],
            "pairs": row["pairs"],
        }

    status = stage_status()
    status["R5-D0"] = {"go": False, "selfstress_family_stop": True}
    status["R6-A0"] = {"frozen": True}
    status["R6-A1"] = {
        "u1_px": statsx["u1_lowconf_fraction"],
        "b0_licensed": b0_licensed,
        "fork": fork,
    }
    payload = {
        "stage": "R6-A1",
        "scientific_result": True,
        "neural_probe": False,
        "retrained": False,
        "planner_changed": False,
        "delta_method": False,
        "z_low": Z_LOW,
        "prereg_sha256": _prereg_sha256(),
        "a0_prereg_sha256": a0.get("prereg_sha256"),
        "d0_prereg_sha256": d0.get("prereg_sha256"),
        "d0_actions_match": d0_match,
        "table_p0": tables["p0"],
        "table_px": tables["px"],
        "stats_p0": stats0,
        "stats_px": statsx,
        "cases_px": cases,
        "px_all_errors_lowconf": px_all_low,
        "lam2_px_confident_wrong": lam2_confident_wrong,
        "b0_licensed": b0_licensed,
        "fork": fork,
        "note": "nested-LOO RMS in J-margin space; no new planner",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
