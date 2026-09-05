"""R5-D0: executed regret of h^S vs (h^S, X) world-model planners. No Adam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r4_c2 import apply_ridge, fit_ridge
from .r5_d0_preflight import ACTIONS, _argmin_unique, task_loss, task_loss_y
from .r5_i0_selfstress import LAMBDAS, simulate_selfstress
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R5/R5_D0_PREREG.md"
RIDGE = 1.0e-3
SHUFFLE_SEED = 0
REL_MIN = 0.05


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R5-D0 locked until D0-preflight has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R5-D0-preflight":
        raise RuntimeError("R5-D0 locked: preflight summary missing")
    if not payload.get("r5_d0_preflight_pass"):
        raise RuntimeError("R5-D0 locked: preflight did not pass")
    return payload


def _oracle_bank() -> dict[float, dict[str, Any]]:
    bank: dict[float, dict[str, Any]] = {}
    for lam in LAMBDAS:
        item: dict[str, Any] = {"lam": float(lam), "y": {}, "j": {}, "hs": None, "x": None}
        js = []
        for name, force in ACTIONS:
            trace = simulate_selfstress(lam, f_diag=force)
            item["y"][name] = np.asarray(trace["y_future"], dtype=np.float64)
            item["j"][name] = task_loss(trace)
            js.append(item["j"][name])
            if name == "mid":
                item["hs"] = np.asarray(trace["hs"], dtype=np.float64)
                item["x"] = float(trace["x_stat"])
        item["a_star"] = _argmin_unique(np.asarray(js, dtype=np.float64))
        bank[float(lam)] = item
    return bank


def _choose(js: dict[str, float]) -> str | None:
    arr = np.asarray([js[name] for name, _ in ACTIONS], dtype=np.float64)
    return _argmin_unique(arr)


def _fit_action_models(
    bank: dict[float, dict[str, Any]],
    train_lams: list[float],
    *,
    x_of: dict[float, float],
    with_x: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    models = {}
    hs = np.stack([bank[lam]["hs"] for lam in train_lams])
    for name, _ in ACTIONS:
        y = np.stack([bank[lam]["y"][name].reshape(-1) for lam in train_lams])
        if with_x:
            x = np.asarray([[x_of[lam]] for lam in train_lams], dtype=np.float64)
            feat = np.concatenate([hs, x], axis=1)
        else:
            feat = hs
        models[name] = fit_ridge(feat, y, ridge=RIDGE)
    return models


def _predict_y(
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    hs: np.ndarray,
    x: float | None,
    y_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    if x is None:
        feat = hs.reshape(1, -1)
    else:
        feat = np.concatenate([hs.reshape(1, -1), np.asarray([[x]], dtype=np.float64)], axis=1)
    return {name: apply_ridge(feat, *pack).reshape(y_shape) for name, pack in models.items()}


def _predict_j(
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    hs: np.ndarray,
    x: float | None,
    y_shape: tuple[int, int],
) -> dict[str, float]:
    pred = _predict_y(models, hs, x, y_shape)
    return {name: task_loss_y(traj) for name, traj in pred.items()}


def run_r5_d0(
    output: str | Path,
    *,
    preflight_summary: str | Path = "runs/r5_d0_preflight/formal/summary.json",
) -> dict[str, Any]:
    pre = _require_preflight(Path(preflight_summary))
    bank = _oracle_bank()
    lams = [float(x) for x in LAMBDAS]
    y_shape = next(iter(bank.values()))["y"]["mid"].shape
    rng = np.random.default_rng(SHUFFLE_SEED)
    x_true = {lam: bank[lam]["x"] for lam in lams}
    perm = rng.permutation(len(lams))
    x_shuf = {lam: x_true[lams[int(perm[i])]] for i, lam in enumerate(lams)}

    rows = []
    for held in lams:
        train = [lam for lam in lams if lam != held]
        m0 = _fit_action_models(bank, train, x_of=x_true, with_x=False)
        mx = _fit_action_models(bank, train, x_of=x_true, with_x=True)
        ms = _fit_action_models(bank, train, x_of=x_shuf, with_x=True)
        hs = bank[held]["hs"]
        j0 = _predict_j(m0, hs, None, y_shape)
        jx = _predict_j(mx, hs, x_true[held], y_shape)
        js = _predict_j(ms, hs, x_shuf[held], y_shape)
        a0 = _choose(j0)
        ax = _choose(jx)
        ash = _choose(js)
        star = bank[held]["a_star"]
        j_star = bank[held]["j"][star] if star is not None else float("nan")

        def regret(action: str | None) -> float:
            if action is None or star is None:
                return float("nan")
            return float(bank[held]["j"][action] - j_star)

        rows.append(
            {
                "lam": held,
                "a_star": star,
                "a0": a0,
                "ax": ax,
                "ashuf": ash,
                "r0": regret(a0),
                "rx": regret(ax),
                "rshuf": regret(ash),
            }
        )

    r0 = np.asarray([row["r0"] for row in rows], dtype=np.float64)
    rx = np.asarray([row["rx"] for row in rows], dtype=np.float64)
    rs = np.asarray([row["rshuf"] for row in rows], dtype=np.float64)
    mean0 = float(np.mean(r0))
    meanx = float(np.mean(rx))
    means = float(np.mean(rs))
    rel01 = (mean0 - meanx) / (abs(mean0) + 1.0e-12)
    rels = (means - meanx) / (abs(means) + 1.0e-12)
    acc0 = float(np.mean([row["a0"] == row["a_star"] for row in rows]))
    accx = float(np.mean([row["ax"] == row["a_star"] for row in rows]))
    differ = float(np.mean([row["ax"] != row["a0"] for row in rows]))
    acc0_zero = float(rows[0]["a0"] == rows[0]["a_star"])
    accx_zero = float(rows[0]["ax"] == rows[0]["a_star"])
    rest = rows[1:]
    acc0_rest = float(np.mean([row["a0"] == row["a_star"] for row in rest]))
    accx_rest = float(np.mean([row["ax"] == row["a_star"] for row in rest]))
    h1 = bool(meanx < mean0 and rel01 > REL_MIN)
    h2 = bool(accx > acc0)
    h3 = bool(differ > 0.0)
    h4 = bool(meanx < means and rels > REL_MIN)
    go = bool(h1 and h2 and h3 and h4)
    status = stage_status()
    status["R4"] = {"family_stop": True}
    status["R5-contact"] = {"paused": True}
    status["R5-I0-SELFSTRESS"] = {"r5_i0_selfstress_pass": False, "frozen": True}
    status["R5-I1-SELFSTRESS"] = {"go": True, "frozen": True}
    status["R5-D0-preflight"] = {"pass": True, "frozen": True}
    status["R5-D0"] = {"go": go, "selfstress_family_stop": (not go)}
    payload = {
        "stage": "R5-D0",
        "scientific_result": True,
        "neural_probe": False,
        "lookup_classifier": False,
        "i0_pass_frozen": False,
        "i1_go_frozen": True,
        "preflight_pass_frozen": True,
        "prereg_sha256": _prereg_sha256(),
        "preflight_prereg_sha256": pre.get("prereg_sha256"),
        "split": "loo_lambda",
        "rows": rows,
        "mean_r0": mean0,
        "mean_rx": meanx,
        "mean_rshuf": means,
        "rel_improve_vs_p0": rel01,
        "rel_improve_vs_shuffle": rels,
        "acc0": acc0,
        "accx": accx,
        "acc0_lam0": acc0_zero,
        "accx_lam0": accx_zero,
        "acc0_lam_ge2": acc0_rest,
        "accx_lam_ge2": accx_rest,
        "p_action_differ": differ,
        "rel_min": REL_MIN,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "r5_d0_go": go,
        "selfstress_family_stop": (not go),
        "note": "world-model then J; executed regret on true rollouts",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
