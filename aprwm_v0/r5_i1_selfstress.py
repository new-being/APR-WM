"""R5-I1-SELFSTRESS: I(X; Y_future | h^S). Linear ridge only. No Adam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r4_c2 import apply_ridge, fit_ridge, mse
from .r5_i0_selfstress import LAMBDAS, simulate_selfstress
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R5/R5_I1_SELFSTRESS_PREREG.md"
TRAIN_LAM = (0.0, 4.0, 8.0, 12.0)
HELD_LAM = (2.0, 6.0, 10.0)
Y_IDX = (0, 1, 2, 4)
L0_MIN = 1.0e-6
DELTA_REL_MIN = 0.05
SHUFFLE_GAP_MIN = 0.05
RIDGE = 1.0e-3
SHUFFLE_SEED = 0


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_i0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R5-I1-SELFSTRESS locked until R5-I0-SELFSTRESS has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R5-I0-SELFSTRESS":
        raise RuntimeError("R5-I1-SELFSTRESS locked: I0 summary is not selfstress")
    if payload.get("r5_i0_selfstress_pass"):
        raise RuntimeError("R5-I1 is for the registered I0 scalar-C failure; I0 PASS is unexpected")
    return payload


def _y_vec(trace: dict[str, Any]) -> np.ndarray:
    y = np.asarray(trace["y_future"], dtype=np.float64)
    return y[:, list(Y_IDX)].reshape(-1)


def _pack(traces: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hs = np.stack([np.asarray(item["hs"], dtype=np.float64) for item in traces])
    x = np.asarray([[item["x_stat"]] for item in traces], dtype=np.float64)
    y = np.stack([_y_vec(item) for item in traces])
    return hs, x, y


def _zscore_y(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) < 1.0e-8, 1.0, train.std(axis=0))
    return (train - mean) / std, (test - mean) / std


def _loss(features: np.ndarray, yz: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    pred = apply_ridge(features, weights, mean, std)
    return mse(pred, yz)


def run_r5_i1_selfstress(
    output: str | Path,
    *,
    i0_summary: str | Path = "runs/r5_i0_selfstress/formal/summary.json",
) -> dict[str, Any]:
    i0 = _require_i0(Path(i0_summary))
    traces = {float(item["lam"]): item for item in (simulate_selfstress(lam) for lam in LAMBDAS)}
    train = [traces[lam] for lam in TRAIN_LAM]
    held = [traces[lam] for lam in HELD_LAM]
    hs_tr, x_tr, y_tr = _pack(train)
    hs_te, x_te, y_te = _pack(held)
    yz_tr, yz_te = _zscore_y(y_tr, y_te)
    feat0_tr, feat0_te = hs_tr, hs_te
    featx_tr = np.concatenate([hs_tr, x_tr], axis=1)
    featx_te = np.concatenate([hs_te, x_te], axis=1)
    w0, m0, s0 = fit_ridge(feat0_tr, yz_tr, ridge=RIDGE)
    wx, mx, sx = fit_ridge(featx_tr, yz_tr, ridge=RIDGE)
    l0 = _loss(feat0_te, yz_te, w0, m0, s0)
    lx = _loss(featx_te, yz_te, wx, mx, sx)
    rng = np.random.default_rng(SHUFFLE_SEED)
    x_shuf = x_te[rng.permutation(len(x_te))]
    feat_shuf = np.concatenate([hs_te, x_shuf], axis=1)
    l_shuf = _loss(feat_shuf, yz_te, wx, mx, sx)
    delta = l0 - lx
    delta_shuf = l0 - l_shuf
    rel = delta / (l0 + 1.0e-12)
    rel_shuf = delta_shuf / (l0 + 1.0e-12)
    h_var = bool(l0 > L0_MIN)
    h_gain = bool(rel > DELTA_REL_MIN)
    h_shuffle = bool(rel > rel_shuf + SHUFFLE_GAP_MIN)
    go = bool(h_var and h_gain and h_shuffle)
    status = stage_status()
    status["R4"] = {"family_stop": True}
    status["R5-contact"] = {"paused": True}
    status["R5-I0-SELFSTRESS"] = {"r5_i0_selfstress_pass": False, "frozen": True}
    status["R5-I1-SELFSTRESS"] = {"go": go, "selfstress_family_stop": (not go)}
    payload = {
        "stage": "R5-I1-SELFSTRESS",
        "scientific_result": True,
        "neural_probe": False,
        "i0_pass_frozen": False,
        "prereg_sha256": _prereg_sha256(),
        "i0_prereg_sha256": i0.get("prereg_sha256"),
        "train_lam": list(TRAIN_LAM),
        "held_lam": list(HELD_LAM),
        "y_idx": list(Y_IDX),
        "l_y_p0": l0,
        "l_y_px": lx,
        "l_y_shuffle": l_shuf,
        "delta_y": delta,
        "delta_y_rel": rel,
        "delta_y_shuffle_rel": rel_shuf,
        "delta_rel_min": DELTA_REL_MIN,
        "shuffle_gap_min": SHUFFLE_GAP_MIN,
        "h_var": h_var,
        "h_gain": h_gain,
        "h_shuffle": h_shuffle,
        "r5_i1_selfstress_go": go,
        "selfstress_family_stop": (not go),
        "note": "ridge only; Y has no X; I0 scalar C untouched",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
