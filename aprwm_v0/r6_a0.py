"""R6-A0: frozen D0 action-margin / cost-error audit. No training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r5_d0 import _choose, _fit_action_models, _oracle_bank, _predict_y
from .r5_d0_preflight import ACTIONS, BETA_LIM, BETA_U, Q_LIM, Y_STAR, task_loss_y
from .r5_i0_selfstress import LAMBDAS
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R6/R6_A0_PREREG.md"


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_d0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R6-A0 locked until R5-D0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R5-D0":
        raise RuntimeError("R6-A0 locked: D0 summary missing")
    return payload


def grad_j(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    g = np.zeros_like(y)
    steps = max(len(y), 1)
    q = y[:, 0]
    u = y[:, 3]
    g[-1, 0] += 2.0 * (float(q[-1]) - Y_STAR)
    g[:, 3] += (2.0 * BETA_U / steps) * u
    excess = np.abs(q) - Q_LIM
    active = excess > 0.0
    g[active, 0] += (2.0 * BETA_LIM / steps) * excess[active] * np.sign(q[active])
    return g


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    if ss_tot < 1.0e-18:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _audit_one(
    *,
    bank: dict[float, dict[str, Any]],
    held: float,
    pred_y: dict[str, np.ndarray],
    hat_a: str | None,
) -> dict[str, Any]:
    star = bank[held]["a_star"]
    names = [name for name, _ in ACTIONS]
    j_true = {name: float(bank[held]["j"][name]) for name in names}
    j_hat = {name: task_loss_y(pred_y[name]) for name in names}
    e_j = {name: j_hat[name] - j_true[name] for name in names}
    competitors = [name for name in names if name != star]
    margins = {name: j_true[name] - j_true[star] for name in competitors}
    nearest = min(competitors, key=lambda n: margins[n])
    rho = {}
    hat_m = {}
    for name in competitors:
        delta_e = e_j[name] - e_j[star]
        hat_m[name] = margins[name] + delta_e
        rho[name] = delta_e / (margins[name] + 1.0e-18)
    ly = float(np.mean([_mse(pred_y[name], bank[held]["y"][name]) for name in names]))
    g_dot = {}
    dpar = {}
    dperp = {}
    for name in names:
        y = bank[held]["y"][name]
        dy = pred_y[name] - y
        g = grad_j(y).reshape(-1)
        dvec = dy.reshape(-1)
        gdot = float(np.dot(g, dvec))
        g2 = float(np.dot(g, g))
        parallel = (gdot / g2) * g if g2 > 1.0e-18 else np.zeros_like(g)
        perp = dvec - parallel
        g_dot[name] = gdot
        dpar[name] = float(np.linalg.norm(parallel))
        dperp[name] = float(np.linalg.norm(perp))
    viol = [name for name in competitors if hat_m[name] < 0.0]
    preserved = hat_a == star
    explained = True
    if preserved:
        explained = len(viol) == 0
    elif hat_a in hat_m:
        explained = hat_m[hat_a] < 0.0
    else:
        explained = False
    return {
        "lam": held,
        "a_star": star,
        "hat_a": hat_a,
        "preserved": preserved,
        "d1_explained": explained,
        "m_nearest": margins[nearest],
        "nearest": nearest,
        "delta_e_nearest": e_j[nearest] - e_j[star],
        "rho_nearest": rho[nearest],
        "hat_m_nearest": hat_m[nearest],
        "violations": viol,
        "l_y": ly,
        "e_j": e_j,
        "m": margins,
        "rho": rho,
        "hat_m": hat_m,
        "g_dot": g_dot,
        "dpar": dpar,
        "dperp": dperp,
        "mean_abs_gdot": float(np.mean([abs(v) for v in g_dot.values()])),
        "mean_dperp": float(np.mean(list(dperp.values()))),
    }


def run_r6_a0(
    output: str | Path,
    *,
    d0_summary: str | Path = "runs/r5_d0/formal/summary.json",
) -> dict[str, Any]:
    d0 = _require_d0(Path(d0_summary))
    bank = _oracle_bank()
    lams = [float(x) for x in LAMBDAS]
    y_shape = next(iter(bank.values()))["y"]["mid"].shape
    x_true = {lam: bank[lam]["x"] for lam in lams}
    tables = {"p0": [], "px": []}
    lin_pred = []
    lin_true = []
    for held in lams:
        train = [lam for lam in lams if lam != held]
        m0 = _fit_action_models(bank, train, x_of=x_true, with_x=False)
        mx = _fit_action_models(bank, train, x_of=x_true, with_x=True)
        hs = bank[held]["hs"]
        y0 = _predict_y(m0, hs, None, y_shape)
        yx = _predict_y(mx, hs, x_true[held], y_shape)
        a0 = _choose({name: task_loss_y(traj) for name, traj in y0.items()})
        ax = _choose({name: task_loss_y(traj) for name, traj in yx.items()})
        row0 = _audit_one(bank=bank, held=held, pred_y=y0, hat_a=a0)
        rowx = _audit_one(bank=bank, held=held, pred_y=yx, hat_a=ax)
        d0_row = next(item for item in d0["rows"] if float(item["lam"]) == held)
        row0["d0_hat_a"] = d0_row["a0"]
        rowx["d0_hat_a"] = d0_row["ax"]
        row0["d0_match"] = row0["hat_a"] == d0_row["a0"]
        rowx["d0_match"] = rowx["hat_a"] == d0_row["ax"]
        tables["p0"].append(row0)
        tables["px"].append(rowx)
        for pred_y in (y0, yx):
            for name, _ in ACTIONS:
                y = bank[held]["y"][name]
                dy = pred_y[name] - y
                lin_pred.append(float(np.dot(grad_j(y).reshape(-1), dy.reshape(-1))))
                lin_true.append(task_loss_y(pred_y[name]) - float(bank[held]["j"][name]))

    def _mean_ly(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([row["l_y"] for row in rows]))

    def _n_err(rows: list[dict[str, Any]]) -> int:
        return int(sum(not row["preserved"] for row in rows))

    def _n_viol(rows: list[dict[str, Any]]) -> int:
        return int(sum(len(row["violations"]) for row in rows))

    def _mean_abs_de(rows: list[dict[str, Any]]) -> float:
        return float(np.mean([abs(row["delta_e_nearest"]) for row in rows]))

    ly0 = _mean_ly(tables["p0"])
    lyx = _mean_ly(tables["px"])
    d0_match = bool(all(row["d0_match"] for row in tables["p0"] + tables["px"]))
    d1 = bool(all(row["d1_explained"] for row in tables["p0"] + tables["px"]))
    d2 = bool(lyx < ly0 and (_n_viol(tables["px"]) > _n_viol(tables["p0"]) or _mean_abs_de(tables["px"]) > _mean_abs_de(tables["p0"])))
    r2 = _r2(np.asarray(lin_pred), np.asarray(lin_true))
    status = stage_status()
    status["R5-D0"] = {"go": False, "selfstress_family_stop": True}
    status["R6-A0"] = {"d1": d1, "d2": d2, "d3_r2": r2}
    payload = {
        "stage": "R6-A0",
        "scientific_result": True,
        "neural_probe": False,
        "retrained": False,
        "prereg_sha256": _prereg_sha256(),
        "d0_prereg_sha256": d0.get("prereg_sha256"),
        "table_p0": tables["p0"],
        "table_px": tables["px"],
        "mean_l_y_p0": ly0,
        "mean_l_y_px": lyx,
        "n_rank_err_p0": _n_err(tables["p0"]),
        "n_rank_err_px": _n_err(tables["px"]),
        "n_margin_viol_p0": _n_viol(tables["p0"]),
        "n_margin_viol_px": _n_viol(tables["px"]),
        "mean_abs_de_nearest_p0": _mean_abs_de(tables["p0"]),
        "mean_abs_de_nearest_px": _mean_abs_de(tables["px"]),
        "mean_abs_gdot_p0": float(np.mean([row["mean_abs_gdot"] for row in tables["p0"]])),
        "mean_abs_gdot_px": float(np.mean([row["mean_abs_gdot"] for row in tables["px"]])),
        "mean_dperp_p0": float(np.mean([row["mean_dperp"] for row in tables["p0"]])),
        "mean_dperp_px": float(np.mean([row["mean_dperp"] for row in tables["px"]])),
        "d0_actions_match": d0_match,
        "d1_attribution": d1,
        "d2_metric_mismatch": d2,
        "d3_r2_linear": r2,
        "d3_nonlinear_flag": bool(r2 < 0.5),
        "note": "offline audit of frozen D0 maps; no new planner",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
