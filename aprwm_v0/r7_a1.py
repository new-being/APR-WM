"""R7-A1: frozen A0 certificate on the saturated plant. No model upgrade."""

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
    EPS,
    POLY_DEG,
    SHUFFLE_SEED,
    _eval_split,
    _failure_locus,
)
from .r7_p1 import ALPHAS as P1_ALPHAS
from .r7_p1 import F_MAX, U_ALT, U_DEFAULT, simulate_saturated, task_loss
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_A1_PREREG.md"
ALPHAS_FIT = (0.45, 0.7, 0.95, 1.2, 1.45, 1.65, 1.9, 2.15, 2.4, 2.65, 2.85, 3.1, 3.35, 3.6)
ALPHAS_CAL = (0.55, 0.75, 1.0, 1.25, 1.5, 1.75, 1.95, 2.2, 2.45, 2.7, 2.95, 3.15, 3.4, 3.65)
ALPHAS_TEST = (
    0.6,
    0.65,
    0.85,
    0.9,
    1.05,
    1.15,
    1.3,
    1.35,
    1.55,
    1.6,
    1.8,
    1.85,
    2.05,
    2.1,
    2.25,
    2.35,
    2.5,
    2.55,
    2.75,
    2.8,
    3.0,
    3.05,
    3.25,
    3.3,
    3.45,
    3.55,
    3.7,
    3.75,
)


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_p1(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-A1 locked until R7-P1 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-P1":
        raise RuntimeError("R7-A1 locked: P1 summary missing")
    if not payload.get("r7_p1_pass"):
        raise RuntimeError("R7-A1 locked: P1 did not pass")
    return payload


def oracle_row(alpha: float) -> dict[str, Any]:
    mid = simulate_saturated(alpha, U_DEFAULT)
    cons = simulate_saturated(alpha, U_ALT)
    return {
        "alpha": float(alpha),
        "x": float(mid["x"]),
        "A": float(task_loss(mid) - task_loss(cons)),
        "j_default": task_loss(mid),
        "j_alt": task_loss(cons),
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


def _pattern(aligned: dict[str, Any]) -> str:
    if not aligned["g_safe"]:
        return "certificate_invalid"
    if aligned["g_recall"]:
        return "robust_positive_certification"
    return "safe_but_over_conservative"


def run_r7_a1(
    output: str | Path,
    *,
    p1_summary: str | Path = "runs/r7_p1/formal/summary.json",
) -> dict[str, Any]:
    p1 = _require_p1(Path(p1_summary))
    p1_set = {round(float(x), 4) for x in P1_ALPHAS}
    for split in (ALPHAS_FIT, ALPHAS_CAL, ALPHAS_TEST):
        if any(round(float(a), 4) in p1_set for a in split):
            raise RuntimeError("R7-A1 split collides with P1 grid")
    fit = _pack(ALPHAS_FIT)
    cal = _pack(ALPHAS_CAL)
    test = _pack(ALPHAS_TEST)
    aligned = _eval_split(fit=fit, cal=cal, test=test)
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = _eval_split(
        fit=_pack(ALPHAS_FIT, x_perm=rng.permutation(fit["x"])),
        cal=_pack(ALPHAS_CAL, x_perm=rng.permutation(cal["x"])),
        test=_pack(ALPHAS_TEST, x_perm=rng.permutation(test["x"])),
    )
    g_ctrl = bool(aligned["voi"] > shuf["voi"])
    locus = _failure_locus(aligned, g_ctrl)
    go = locus is None
    pattern = _pattern(aligned)
    status = stage_status()
    status["R7-P1"] = {"pass": True, "frozen": True}
    status["R7-A1"] = {"go": go, "pattern": pattern, "failure_locus": locus}
    payload = {
        "stage": "R7-A1",
        "scientific_result": True,
        "neural_probe": False,
        "model_upgraded": False,
        "f_max": F_MAX,
        "poly_deg": POLY_DEG,
        "delta": DELTA,
        "eps": EPS,
        "n_test": len(ALPHAS_TEST),
        "prereg_sha256": _prereg_sha256(),
        "p1_prereg_sha256": p1.get("prereg_sha256"),
        "aligned": aligned,
        "shuffled": shuf,
        "g_cert": aligned["g_cert"],
        "g_safe": aligned["g_safe"],
        "g_recall": aligned["g_recall"],
        "g_voi": aligned["g_voi"],
        "g_ctrl": g_ctrl,
        "pattern": pattern,
        "r7_a1_go": go,
        "failure_locus": locus,
        "note": "frozen A0 certificate on saturated plant; P1 grid unused",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
