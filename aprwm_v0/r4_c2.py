"""R4-C2: I(X; C | h^S) with continuous pair-level future-macro C.

Does not retune I0 or a^diag. Does not use X in the label.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r4_i0 import R4I0Config
from .r4_c1 import _collect
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R4/R4_C2_PREREG.md"
DELTA_MIN = 0.05
RIDGE = 1.0e-3
STD_C_MIN = 1.0e-6
TRAIN_Q0 = tuple(float(x) for x in np.linspace(-0.0008, 0.0008, 17))
HELD_Q0 = (-0.0015, -0.00135, -0.0012, -0.00105, 0.00105, 0.0012, 0.00135, 0.0015)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zscore_fit(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < STD_C_MIN:
        std = 1.0
    return mean, std


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) / std


def fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) < 1.0e-8, 1.0, x.std(axis=0))
    z = (x - mean) / std
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    dim = design.shape[1]
    gram = design.T @ design + ridge * np.eye(dim)
    weights = np.linalg.solve(gram, design.T @ y)
    return weights, mean, std


def apply_ridge(features: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (np.asarray(features, dtype=np.float64) - mean) / std
    design = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    return design @ weights


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def _rows(pairs: list[dict[str, Any]], *, with_x: bool, canon: bool, shuffle: bool, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    extras = []
    for item in pairs:
        for side in ("a", "b"):
            h = item[f"h_{side}"]
            x = item["x_b_canon"] if canon and side == "b" else item[f"x_{side}"]
            feat = np.concatenate([h, x], axis=0) if with_x else h
            feats.append(feat)
            labels.append(float(item["c"]))
            extras.append(x if with_x else None)
    features = np.stack(feats)
    target = np.asarray(labels, dtype=np.float64)
    if with_x and shuffle:
        generator = rng if rng is not None else np.random.default_rng(0)
        order = generator.permutation(len(features))
        h_dim = pairs[0]["h_a"].shape[0]
        features = features.copy()
        features[:, h_dim:] = features[order, h_dim:]
    return features, target


def _require(i0: Path, i1: Path, c1: Path) -> None:
    for path, name in ((i0, "I0"), (i1, "I1"), (c1, "C1")):
        if not path.is_file():
            raise RuntimeError(f"R4-C2 locked until R4-{name} has been run")
    i0_p = json.loads(i0.read_text(encoding="utf-8"))
    i1_p = json.loads(i1.read_text(encoding="utf-8"))
    c1_p = json.loads(c1.read_text(encoding="utf-8"))
    if not i0_p.get("r4_i0_pass") or i0_p.get("protocol") != "stiffness_swap_v2":
        raise RuntimeError("R4-C2 locked: I0 v2 did not pass")
    if not i1_p.get("r4_i1_pass"):
        raise RuntimeError("R4-C2 locked: I1 did not pass")
    if not c1_p.get("f0"):
        raise RuntimeError("R4-C2 locked: C1 F0 is false")


def run_r4_c2(
    output: str | Path,
    *,
    i0_summary: str | Path,
    i1_summary: str | Path,
    c1_summary: str | Path,
) -> dict[str, Any]:
    _require(Path(i0_summary), Path(i1_summary), Path(c1_summary))
    config = R4I0Config()
    train = _collect(TRAIN_Q0, config)
    held = _collect(HELD_Q0, config)
    train_c = np.asarray([item["c"] for item in train], dtype=np.float64)
    held_c = np.asarray([item["c"] for item in held], dtype=np.float64)
    std_c = float(train_c.std())
    degenerate = bool(std_c < STD_C_MIN)
    mean_c, scale_c = zscore_fit(train_c)
    rng = np.random.default_rng(21)
    payload_f: dict[str, Any] = {
        "evaluated": False,
        "degenerate_c": degenerate,
        "loss_hs": math.nan,
        "loss_hs_x": math.nan,
        "loss_hs_x_canon": math.nan,
        "loss_hs_x_shuffle": math.nan,
        "delta_c": math.nan,
        "delta_c_canon": math.nan,
        "delta_c_shuffle": math.nan,
    }
    if not degenerate:
        z_tr = zscore(train_c, mean_c, scale_c)
        # pair C is shared; expand to A/B rows
        hs_tr, c_tr_rows = _rows(train, with_x=False, canon=False, shuffle=False, rng=rng)
        hx_tr, _ = _rows(train, with_x=True, canon=False, shuffle=False, rng=rng)
        hc_tr, _ = _rows(train, with_x=True, canon=True, shuffle=False, rng=rng)
        hs_sh, _ = _rows(train, with_x=True, canon=False, shuffle=True, rng=rng)
        z_rows = np.repeat(z_tr, 2)
        w0, m0, s0 = fit_ridge(hs_tr, z_rows)
        wx, mx, sx = fit_ridge(hx_tr, z_rows)
        wc, mc, sc = fit_ridge(hc_tr, z_rows)
        ws, ms, ss = fit_ridge(hs_sh, z_rows)
        hs_te, _ = _rows(held, with_x=False, canon=False, shuffle=False, rng=rng)
        hx_te, _ = _rows(held, with_x=True, canon=False, shuffle=False, rng=rng)
        hc_te, _ = _rows(held, with_x=True, canon=True, shuffle=False, rng=rng)
        hx_shu, _ = _rows(held, with_x=True, canon=False, shuffle=True, rng=rng)
        z_te = np.repeat(zscore(held_c, mean_c, scale_c), 2)
        loss0 = mse(apply_ridge(hs_te, w0, m0, s0), z_te)
        lossx = mse(apply_ridge(hx_te, wx, mx, sx), z_te)
        lossc = mse(apply_ridge(hc_te, wc, mc, sc), z_te)
        losss = mse(apply_ridge(hx_shu, ws, ms, ss), z_te)
        payload_f = {
            "evaluated": True,
            "degenerate_c": False,
            "loss_hs": loss0,
            "loss_hs_x": lossx,
            "loss_hs_x_canon": lossc,
            "loss_hs_x_shuffle": losss,
            "delta_c": float(loss0 - lossx),
            "delta_c_canon": float(loss0 - lossc),
            "delta_c_shuffle": float(loss0 - losss),
        }
    delta = payload_f["delta_c"]
    passed = bool(
        payload_f["evaluated"] and math.isfinite(delta) and delta > DELTA_MIN
    )
    family_stop = not passed
    status = stage_status()
    status["R4-C1"] = {"f0": True, "r4_c1_go": False, "frozen": True}
    status["R4-C2"] = {"r4_c2_go": passed, "delta_c": delta, "r4_family_stop": family_stop}
    payload = {
        "stage": "R4-C2",
        "scientific_result": True,
        "prereg_sha256": _prereg_sha256(),
        "i0_frozen": True,
        "probe_frozen": True,
        "label_from_x": False,
        "pair_shared_c": True,
        "train_q0": list(TRAIN_Q0),
        "held_q0": list(HELD_Q0),
        "n_train_pairs": len(train),
        "n_held_pairs": len(held),
        "train_c": train_c.tolist(),
        "held_c": held_c.tolist(),
        "train_mean_c": float(train_c.mean()),
        "train_std_c": std_c,
        "held_mean_c": float(held_c.mean()),
        "held_std_c": float(held_c.std()),
        "c_z_mean": mean_c,
        "c_z_std": scale_c,
        "delta_min": DELTA_MIN,
        "matched_at_star_held": all(item["matched_at_star"] for item in held),
        **payload_f,
        "r4_c2_go": passed,
        "r4_family_stop": family_stop,
        "note": "z-scored pair C; A/B share C; more q0 only",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
