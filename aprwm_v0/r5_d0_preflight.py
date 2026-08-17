"""R5-D0-preflight: oracle action ranking vs preload. No planner, no Adam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r5_i0_selfstress import LAMBDAS, simulate_selfstress
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R5_D0_PREFLIGHT_PREREG.md"
ACTIONS = (("cons", 0.4), ("mid", 1.0), ("agg", 2.5))
Y_STAR = 0.010
BETA_U = 1.0e-5
BETA_LIM = 10.0
Q_LIM = 0.045
TIE_EPS = 1.0e-12


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_i1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R5-D0-preflight locked until R5-I1-SELFSTRESS has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R5-I1-SELFSTRESS":
        raise RuntimeError("R5-D0-preflight locked: I1 summary is not selfstress")
    if not payload.get("r5_i1_selfstress_go"):
        raise RuntimeError("R5-D0-preflight locked: I1 GO is false")
    return payload


def task_loss(trace: dict[str, Any]) -> float:
    y = np.asarray(trace["y_future"], dtype=np.float64)
    q = y[:, 0]
    u = y[:, 3]
    pos = float((float(trace["q_end"]) - Y_STAR) ** 2)
    effort = BETA_U * float(np.mean(u**2))
    excess = np.maximum(np.abs(q) - Q_LIM, 0.0)
    limit = BETA_LIM * float(np.mean(excess**2))
    return pos + effort + limit


def _argmin_unique(js: np.ndarray) -> str | None:
    best = float(np.min(js))
    hits = [ACTIONS[i][0] for i, val in enumerate(js) if abs(float(val) - best) <= TIE_EPS]
    if len(hits) != 1:
        return None
    return hits[0]


def run_r5_d0_preflight(
    output: str | Path,
    *,
    i1_summary: str | Path = "runs/r5_i1_selfstress/formal/summary.json",
) -> dict[str, Any]:
    i1 = _require_i1(Path(i1_summary))
    table = []
    stars = []
    for lam in LAMBDAS:
        row: dict[str, Any] = {"lam": float(lam)}
        js = []
        for name, force in ACTIONS:
            trace = simulate_selfstress(lam, f_diag=force)
            loss = task_loss(trace)
            row[f"j_{name}"] = loss
            row[f"q_end_{name}"] = float(trace["q_end"])
            js.append(loss)
        js_arr = np.asarray(js, dtype=np.float64)
        star = _argmin_unique(js_arr)
        row["a_star"] = star
        table.append(row)
        if star is not None:
            stars.append(star)
    unique = tuple(dict.fromkeys(stars))
    g_flip = bool(len(set(stars)) >= 2)
    passed = bool(g_flip)
    status = stage_status()
    status["R4"] = {"family_stop": True}
    status["R5-contact"] = {"paused": True}
    status["R5-I0-SELFSTRESS"] = {"r5_i0_selfstress_pass": False, "frozen": True}
    status["R5-I1-SELFSTRESS"] = {"go": True, "frozen": True}
    status["R5-D0-preflight"] = {
        "pass": passed,
        "selfstress_family_stop": (not passed),
    }
    payload = {
        "stage": "R5-D0-preflight",
        "scientific_result": True,
        "neural_probe": False,
        "planner": False,
        "i0_pass_frozen": False,
        "i1_go_frozen": True,
        "prereg_sha256": _prereg_sha256(),
        "i1_prereg_sha256": i1.get("prereg_sha256"),
        "actions": {name: force for name, force in ACTIONS},
        "y_star": Y_STAR,
        "beta_u": BETA_U,
        "beta_lim": BETA_LIM,
        "q_lim": Q_LIM,
        "table": table,
        "a_star_sequence": [row["a_star"] for row in table],
        "unique_a_star": list(unique),
        "g_flip": g_flip,
        "r5_d0_preflight_pass": passed,
        "selfstress_family_stop": (not passed),
        "note": "oracle J only; no P0/PX planner; I0 C untouched",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
