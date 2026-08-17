"""R4-C1: future-macro consequence warrant. No X in the label. No encoder."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r4_i0 import R4I0Config, mechanics_matched, simulate_pattern
from .r4_i1 import (
    AUROC_LEAK_FAIL,
    AUROC_PASS_MAX,
    GAIN_MIN,
    HISTORY_W,
    TRAIN_Q0,
    _hs,
    _history,
    _tactile,
    apply_linear,
    auroc,
    fit_linear,
)
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R4_C1_PREREG.md"
FT_PROBE = 4.8
PROBE_S = 0.40
C_MIN = 1.0e-3
Y_KEYS = ("q", "v", "fn", "ft", "fx", "fy", "fz", "mx", "my", "mz")
F0_HELD_Q0 = (-0.0012, 0.0012, -0.0015, 0.0015)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_c(y_a: np.ndarray, y_b: np.ndarray) -> float:
    a = np.asarray(y_a, dtype=np.float64).reshape(-1)
    b = np.asarray(y_b, dtype=np.float64).reshape(-1)
    denom = 0.5 * float(np.linalg.norm(a)) + 0.5 * float(np.linalg.norm(b)) + 1.0e-12
    return float(np.linalg.norm(a - b) / denom)


def _y_window(rows: list[dict[str, Any]], start: int) -> np.ndarray:
    return np.stack(
        [np.asarray([float(row[key]) for key in Y_KEYS], dtype=np.float64) for row in rows[start:]],
        axis=0,
    )


def _require_locks(i0_path: Path, i1_path: Path, c0_path: Path) -> None:
    if not i0_path.is_file():
        raise RuntimeError("R4-C1 locked until R4-I0 has been run")
    if not i1_path.is_file():
        raise RuntimeError("R4-C1 locked until R4-I1 has been run")
    if not c0_path.is_file():
        raise RuntimeError("R4-C1 locked until R4-C0 has been run")
    i0 = json.loads(i0_path.read_text(encoding="utf-8"))
    i1 = json.loads(i1_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    if not i0.get("r4_i0_pass") or i0.get("protocol") != "stiffness_swap_v2":
        raise RuntimeError("R4-C1 locked: I0 v2 did not pass")
    if not i1.get("r4_i1_pass"):
        raise RuntimeError("R4-C1 locked: I1 did not pass")
    if c0.get("stage") != "R4-C0" or not c0.get("scientific_result"):
        raise RuntimeError("R4-C1 locked: C0 scientific matrix has not been run")


def _pair(q0: float, config: R4I0Config) -> dict[str, Any]:
    left = simulate_pattern(
        "stiff_left", config, q0_block=q0, probe_ft=FT_PROBE, probe_s=PROBE_S
    )
    right = simulate_pattern(
        "stiff_right", config, q0_block=q0, probe_ft=FT_PROBE, probe_s=PROBE_S
    )
    t_star = int(left["n_hold"]) - 1
    a_star, b_star = left["rows"][t_star], right["rows"][t_star]
    matched = mechanics_matched(a_star, b_star)
    cost = relative_c(_y_window(left["rows"], t_star), _y_window(right["rows"], t_star))
    hs_left = [_hs(row) for row in left["rows"][: t_star + 1]]
    hs_right = [_hs(row) for row in right["rows"][: t_star + 1]]
    e_hat = 1.0 if cost > C_MIN else 0.0
    return {
        "q0": float(q0),
        "t_star": t_star,
        "t": float(a_star["t"]),
        "matched_at_star": matched,
        "c": cost,
        "e_future": e_hat,
        "h_a": _history(hs_left, t_star),
        "h_b": _history(hs_right, t_star),
        "x_a": _tactile(a_star),
        "x_b": _tactile(b_star),
        "x_a_canon": _tactile(a_star),
        "x_b_canon": _tactile(b_star, x_flip=True),
    }


def _collect(q0s: tuple[float, ...], config: R4I0Config) -> list[dict[str, Any]]:
    return [_pair(q0, config) for q0 in q0s]


def _probe_matrix(pairs: list[dict[str, Any]], *, with_x: bool, canon: bool) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for item in pairs:
        for side in ("a", "b"):
            h = item[f"h_{side}"]
            if with_x:
                x = item["x_b_canon"] if canon and side == "b" else item[f"x_{side}"]
                feats.append(np.concatenate([h, x], axis=0))
            else:
                feats.append(h)
            labels.append(item["e_future"])
    return np.stack(feats), np.asarray(labels, dtype=np.float64)


def _eval_f1(train: list[dict[str, Any]], held: list[dict[str, Any]]) -> dict[str, Any]:
    y_tr = np.asarray([item["e_future"] for item in train])
    y_te = np.asarray([item["e_future"] for item in held])
    if y_tr.min() == y_tr.max() or y_te.min() == y_te.max():
        return {
            "evaluated": False,
            "degenerate_target": True,
            "auroc_hs": math.nan,
            "auroc_hs_x": math.nan,
            "auroc_gain": math.nan,
            "auroc_hs_x_canon": math.nan,
            "h1_no_leak": False,
            "h2_gain": False,
            "leak_fail": False,
            "f1_pass": False,
        }
    hs_tr, y_rows = _probe_matrix(train, with_x=False, canon=False)
    hx_tr, _ = _probe_matrix(train, with_x=True, canon=False)
    hc_tr, _ = _probe_matrix(train, with_x=True, canon=True)
    hs_te, y_held = _probe_matrix(held, with_x=False, canon=False)
    hx_te, _ = _probe_matrix(held, with_x=True, canon=False)
    hc_te, _ = _probe_matrix(held, with_x=True, canon=True)
    w_s, m_s, s_s = fit_linear(hs_tr, y_rows)
    w_x, m_x, s_x = fit_linear(hx_tr, y_rows)
    w_c, m_c, s_c = fit_linear(hc_tr, y_rows)
    auc_s = auroc(apply_linear(hs_te, w_s, m_s, s_s), y_held)
    auc_x = auroc(apply_linear(hx_te, w_x, m_x, s_x), y_held)
    auc_c = auroc(apply_linear(hc_te, w_c, m_c, s_c), y_held)
    gain = (
        float(auc_x - auc_s) if math.isfinite(auc_s) and math.isfinite(auc_x) else math.nan
    )
    leak_fail = bool(math.isfinite(auc_s) and auc_s >= AUROC_LEAK_FAIL)
    h1 = bool(math.isfinite(auc_s) and auc_s < AUROC_PASS_MAX and not leak_fail)
    h2 = bool(math.isfinite(gain) and gain > GAIN_MIN)
    return {
        "evaluated": True,
        "degenerate_target": False,
        "auroc_hs": auc_s,
        "auroc_hs_x": auc_x,
        "auroc_gain": gain,
        "auroc_hs_x_canon": auc_c,
        "h1_no_leak": h1,
        "h2_gain": h2,
        "leak_fail": leak_fail,
        "f1_pass": bool(h1 and h2),
    }


def run_r4_c1(
    output: str | Path,
    *,
    i0_summary: str | Path,
    i1_summary: str | Path,
    c0_summary: str | Path,
) -> dict[str, Any]:
    _require_locks(Path(i0_summary), Path(i1_summary), Path(c0_summary))
    config = R4I0Config()
    train = _collect(TRAIN_Q0, config)
    held = _collect(F0_HELD_Q0, config)
    train_c = [float(item["c"]) for item in train]
    held_c = [float(item["c"]) for item in held]
    mean_c = float(np.mean(held_c)) if held_c else math.nan
    train_mean_c = float(np.mean(train_c)) if train_c else math.nan
    f0 = bool(math.isfinite(mean_c) and mean_c > C_MIN)
    f1: dict[str, Any] = {
        "evaluated": False,
        "skipped": "F0 failed; R4 family stop",
        "f1_pass": False,
    }
    if f0:
        f1 = _eval_f1(train, held)
    family_stop = not f0
    passed = bool(f0 and f1.get("f1_pass"))
    status = stage_status()
    status["R4-I0"] = {"frozen": True, "r4_i0_pass": True}
    status["R4-I1"] = {"frozen": True, "r4_i1_pass": True}
    status["R4-C0"] = {"frozen": True, "r4_c0_go": False}
    status["R4-C1"] = {
        "f0": f0,
        "f1": bool(f1.get("f1_pass")),
        "r4_c1_go": passed,
        "r4_family_stop": family_stop,
        "verdict": "nuisance" if family_stop else ("consequential" if passed else "consequential_unread"),
    }
    payload = {
        "stage": "R4-C1",
        "scientific_result": True,
        "prereg_sha256": _prereg_sha256(),
        "i0_frozen": True,
        "label_from_x": False,
        "ft_probe": FT_PROBE,
        "probe_s": PROBE_S,
        "c_min": C_MIN,
        "y_keys": list(Y_KEYS),
        "history_w": HISTORY_W,
        "train_q0": list(TRAIN_Q0),
        "held_q0": list(F0_HELD_Q0),
        "n_train": len(train),
        "n_held": len(held),
        "train_c": train_c,
        "train_mean_c": train_mean_c,
        "held_c": held_c,
        "held_mean_c": mean_c,
        "held_matched_at_star": all(item["matched_at_star"] for item in held),
        "f0": f0,
        "f1": f1,
        "r4_c1_go": passed,
        "r4_family_stop": family_stop,
        "verdict": status["R4-C1"]["verdict"],
        "note": "e_future from future macro Y only; F1 skipped if F0 fails",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
