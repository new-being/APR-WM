"""R4-I1: no-leak audit of over-complete h^S on frozen I0 v2 A/B.

Does not retune I0 physics. Does not train an encoder. Does not set C0 GO.
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
from .r4_i0 import D_TAXEL_MIN, R4I0Config, d_taxel, mechanics_matched, simulate_pattern
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R4_I1_PREREG.md"
AUROC_LEAK_FAIL = 0.95
AUROC_PASS_MAX = 0.90
GAIN_MIN = 0.10
HISTORY_W = 8
TRAIN_Q0 = (-0.0008, -0.0004, 0.0, 0.0004, 0.0008)
HELD_Q0 = (-0.0012, 0.0012)
HS_KEYS = (
    "q",
    "v",
    "a",
    "q_finger",
    "v_finger",
    "a_finger",
    "u_finger",
    "tau_motor_finger",
    "tau_motor_block",
    "ft_cmd",
    "fn",
    "ft",
    "fx",
    "fy",
    "fz",
    "mx",
    "my",
    "mz",
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    count = 0.0
    for value in pos:
        count += float(np.sum(neg < value)) + 0.5 * float(np.sum(neg == value))
    return float(count / (len(pos) * len(neg)))


def flip_x(maps: np.ndarray) -> np.ndarray:
    """Left–right taxel flip on the pad x axis (last dim)."""
    return np.flip(np.asarray(maps, dtype=np.float64), axis=2)


def _hs(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in HS_KEYS], dtype=np.float64)


def _tactile(row: dict[str, Any], *, x_flip: bool = False) -> np.ndarray:
    p = np.asarray(row["p"], dtype=np.float64)
    tx = np.asarray(row["tau_x"], dtype=np.float64)
    ty = np.asarray(row["tau_y"], dtype=np.float64)
    if x_flip:
        p, tx, ty = flip_x(p), flip_x(tx), flip_x(ty)
    return np.concatenate([p.reshape(-1), tx.reshape(-1), ty.reshape(-1)], axis=0)


def _history(seq: list[np.ndarray], index: int, width: int = HISTORY_W) -> np.ndarray:
    dim = seq[0].shape[0]
    chunks = []
    for lag in range(width - 1, -1, -1):
        k = index - lag
        chunks.append(seq[k] if k >= 0 else np.zeros(dim, dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def fit_linear(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int = 800,
    ridge: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) < 1.0e-8, 1.0, x.std(axis=0))
    z = (x - mean) / std
    w = np.zeros(z.shape[1] + 1, dtype=np.float64)
    n = max(len(y), 1)
    for _ in range(steps):
        pred = 1.0 / (1.0 + np.exp(-np.clip(z @ w[1:] + w[0], -20, 20)))
        err = pred - y
        w[0] -= 0.2 * float(err.mean())
        w[1:] -= 0.2 * ((z.T @ err) / n + ridge * w[1:])
    return w, mean, std


def apply_linear(
    features: np.ndarray,
    w: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    z = (np.asarray(features, dtype=np.float64) - mean) / std
    return z @ w[1:] + w[0]


def _require_i0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R4-I1 locked until R4-I0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R4-I0" or not payload.get("r4_i0_pass"):
        raise RuntimeError("R4-I1 locked: R4_I0_PASS is false")
    if payload.get("protocol") != "stiffness_swap_v2":
        raise RuntimeError("R4-I1 locked: I0 protocol is not stiffness_swap_v2")
    return payload


def _collect_split(
    q0s: tuple[float, ...],
    config: R4I0Config,
) -> dict[str, Any]:
    hs_list: list[np.ndarray] = []
    hx_list: list[np.ndarray] = []
    hx_flip_list: list[np.ndarray] = []
    hx_canon_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    x_flip_list: list[np.ndarray] = []
    labels: list[float] = []
    n_macro = 0
    n_local = 0
    for q0 in q0s:
        left = simulate_pattern("stiff_left", config, q0_block=q0)
        right = simulate_pattern("stiff_right", config, q0_block=q0)
        hs_left = [_hs(row) for row in left["rows"]]
        hs_right = [_hs(row) for row in right["rows"]]
        for index, (a, b) in enumerate(zip(left["rows"], right["rows"], strict=True)):
            if not mechanics_matched(a, b):
                continue
            n_macro += 1
            if d_taxel(a, b) > D_TAXEL_MIN:
                n_local += 1
            h_a = _history(hs_left, index)
            h_b = _history(hs_right, index)
            x_a = _tactile(a)
            x_b = _tactile(b)
            x_a_flip = _tactile(a, x_flip=True)
            x_b_flip = _tactile(b, x_flip=True)
            for h, x, x_flip, label in (
                (h_a, x_a, x_a_flip, 0.0),
                (h_b, x_b, x_b_flip, 1.0),
            ):
                labels.append(label)
                hs_list.append(h)
                x_list.append(x)
                x_flip_list.append(x_flip)
                hx_list.append(np.concatenate([h, x], axis=0))
                hx_flip_list.append(np.concatenate([h, x_flip], axis=0))
                x_canon = x_flip if label > 0.5 else x
                hx_canon_list.append(np.concatenate([h, x_canon], axis=0))
    return {
        "hs": np.stack(hs_list) if hs_list else np.zeros((0, HISTORY_W * len(HS_KEYS))),
        "hx": np.stack(hx_list) if hx_list else np.zeros((0, 1)),
        "hx_flip": np.stack(hx_flip_list) if hx_flip_list else np.zeros((0, 1)),
        "hx_canon": np.stack(hx_canon_list) if hx_canon_list else np.zeros((0, 1)),
        "x": np.stack(x_list) if x_list else np.zeros((0, 1)),
        "x_flip": np.stack(x_flip_list) if x_flip_list else np.zeros((0, 1)),
        "y": np.asarray(labels, dtype=np.float64),
        "n_macro_matched": n_macro,
        "n_local_split": n_local,
        "n_probe": len(labels),
        "n_traj": len(q0s),
    }


def _eval_probe(train: dict[str, Any], held: dict[str, Any], key: str) -> float:
    w, mean, std = fit_linear(train[key], train["y"])
    return auroc(apply_linear(held[key], w, mean, std), held["y"])


def run_r4_i1(
    output: str | Path,
    *,
    i0_summary: str | Path,
) -> dict[str, Any]:
    i0 = _require_i0(Path(i0_summary))
    config = R4I0Config()
    train = _collect_split(TRAIN_Q0, config)
    held = _collect_split(HELD_Q0, config)
    auc_s = _eval_probe(train, held, "hs")
    auc_x = _eval_probe(train, held, "x")
    auc_sx = _eval_probe(train, held, "hx")
    auc_sx_flip = _eval_probe(train, held, "hx_flip")
    auc_sx_canon = _eval_probe(train, held, "hx_canon")
    auc_x_flip = _eval_probe(train, held, "x_flip")
    gain = (
        float(auc_sx - auc_s) if math.isfinite(auc_s) and math.isfinite(auc_sx) else math.nan
    )
    leak_fail = bool(math.isfinite(auc_s) and auc_s >= AUROC_LEAK_FAIL)
    h1 = bool(math.isfinite(auc_s) and auc_s < AUROC_PASS_MAX and not leak_fail)
    h2 = bool(math.isfinite(gain) and gain > GAIN_MIN)
    passed = bool(h1 and h2)
    status = stage_status()
    status["R4-I0"] = {
        "protocol": "stiffness_swap_v2",
        "r4_i0_pass": True,
        "frozen": True,
    }
    status["R4-I1"] = {
        "r4_i1_pass": passed,
        "auroc_hs": auc_s,
        "auroc_gain": gain,
    }
    status["R4-C0"] = {
        "locked": not passed,
        "reason": "I1 must pass before C0 formal" if not passed else "unlocked to implement C0",
    }
    payload = {
        "stage": "R4-I1",
        "scientific_result": False,
        "prereg_sha256": _prereg_sha256(),
        "i0_protocol": i0.get("protocol"),
        "i0_frozen": True,
        "split": "trajectory_q0",
        "train_q0": list(TRAIN_Q0),
        "held_q0": list(HELD_Q0),
        "history_w": HISTORY_W,
        "hs_keys": list(HS_KEYS),
        "n_train_probe": train["n_probe"],
        "n_held_probe": held["n_probe"],
        "n_train_macro_matched": train["n_macro_matched"],
        "n_held_macro_matched": held["n_macro_matched"],
        "n_train_local_split": train["n_local_split"],
        "n_held_local_split": held["n_local_split"],
        "auroc_hs": auc_s,
        "auroc_x": auc_x,
        "auroc_hs_x": auc_sx,
        "auroc_gain": gain,
        "auroc_x_flip": auc_x_flip,
        "auroc_hs_x_flip": auc_sx_flip,
        "auroc_hs_x_canon": auc_sx_canon,
        "leak_fail_threshold": AUROC_LEAK_FAIL,
        "pass_max_auroc_hs": AUROC_PASS_MAX,
        "gain_min": GAIN_MIN,
        "leak_fail": leak_fail,
        "h1_no_leak": h1,
        "h2_gain": h2,
        "r4_i1_pass": passed,
        "note": (
            "trajectory-split linear probes; flip/canonicalize are diagnostics; "
            "not R4-C0 GO"
        ),
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
