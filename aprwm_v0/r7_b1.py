"""R7-B1: frozen B0 certificate under F_max shift. No adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r7_a0 import DELTA, DELTA_USEFUL, EPS, RECALL_MIN, SHUFFLE_SEED, _json_float
from .r7_b0 import ALPHAS_TEST, predict_a_wm
from .r7_p0 import U_ALT, U_DEFAULT
from .r7_p1 import F_MAX, simulate_saturated, task_loss
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R7/R7_B1_PREREG.md"
F_MAX_TEST = 1.5


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_b0(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("R7-B1 locked until R7-B0 has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "R7-B0":
        raise RuntimeError("R7-B1 locked: B0 summary missing")
    if not payload.get("r7_b0_go"):
        raise RuntimeError("R7-B1 locked: B0 GO is false")
    return payload


def oracle_row(alpha: float, f_max: float) -> dict[str, Any]:
    mid = simulate_saturated(alpha, U_DEFAULT, f_max=f_max)
    alt = simulate_saturated(alpha, U_ALT, f_max=f_max)
    return {
        "alpha": float(alpha),
        "x": float(mid["x"]),
        "A": float(task_loss(mid) - task_loss(alt)),
        "j_default": task_loss(mid),
        "j_alt": task_loss(alt),
        "f_max": float(f_max),
    }


def _pack(alphas: tuple[float, ...], f_max: float, x_perm: np.ndarray | None = None) -> dict[str, Any]:
    rows = [oracle_row(alpha, f_max) for alpha in alphas]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    if x_perm is not None:
        x = np.asarray(x_perm, dtype=np.float64)
    return {
        "alpha": np.asarray([row["alpha"] for row in rows], dtype=np.float64),
        "x": x,
        "A": np.asarray([row["A"] for row in rows], dtype=np.float64),
        "rows": rows,
    }


def _score_frozen(pack: dict[str, Any], coefs: np.ndarray, q: float) -> dict[str, Any]:
    ahat = predict_a_wm(pack["x"], coefs)
    lower = ahat - q
    commit = lower > DELTA
    a_true = pack["A"]
    useful = a_true >= DELTA_USEFUL
    n_commit = int(np.sum(commit))
    n_useful = int(np.sum(useful))
    coverage = float(np.mean(a_true >= lower)) if len(a_true) else 0.0
    n_harmful_commit = int(np.sum(commit & (a_true <= 0.0)))
    if n_commit == 0:
        precision = None
        vacuous = True
        no_harmful = True
    else:
        precision = float(np.mean(a_true[commit] > 0.0))
        vacuous = False
        no_harmful = bool(precision >= 1.0 - 1.0e-12)
    recall = float(np.mean(commit[useful])) if n_useful else 0.0
    voi = float(np.mean(a_true * commit))
    g_cert = bool(coverage + 1.0e-12 >= 1.0 - EPS)
    rows = []
    for i, row in enumerate(pack["rows"]):
        item = dict(row)
        item["x_used"] = float(pack["x"][i])
        item["A_hat"] = _json_float(float(ahat[i]))
        item["L"] = _json_float(float(lower[i]))
        item["commit"] = bool(commit[i])
        item["region"] = (
            "useful" if row["A"] >= DELTA_USEFUL else ("gray" if row["A"] >= 0.0 else "harmful")
        )
        rows.append(item)
    return {
        "q": _json_float(q),
        "coverage": coverage,
        "precision": precision,
        "recall_useful": recall,
        "voi": voi,
        "n_commit": n_commit,
        "n_useful": n_useful,
        "n_harmful_commit": n_harmful_commit,
        "vacuous_abstain": vacuous,
        "g_cert": g_cert,
        "g_safe": bool(no_harmful),
        "g_recall": bool(recall + 1.0e-12 >= RECALL_MIN),
        "g_voi": bool(voi > 0.0),
        "rows": rows,
    }


def _pattern(shifted: dict[str, Any], b0_recall: float) -> str:
    shift_valid = bool(shifted["g_cert"] and shifted["g_safe"])
    if not shift_valid:
        return "certificate_not_shift_valid"
    if float(shifted["recall_useful"]) + 1.0e-12 >= b0_recall:
        return "empirical_shift_robustness"
    return "safe_conservative_degradation"


def run_r7_b1(
    output: str | Path,
    *,
    b0_summary: str | Path = "runs/r7_b0/formal/summary.json",
) -> dict[str, Any]:
    b0 = _require_b0(Path(b0_summary))
    coefs = np.asarray(b0["aligned"]["coefs"], dtype=np.float64)
    q = float(b0["aligned"]["q"])
    b0_recall = float(b0["aligned"]["recall_useful"])
    b0_voi = float(b0["aligned"]["voi"])
    b0_cov = float(b0["aligned"]["coverage"])
    id_pack = _pack(ALPHAS_TEST, F_MAX)
    shift_pack = _pack(ALPHAS_TEST, F_MAX_TEST)
    id_replay = _score_frozen(id_pack, coefs, q)
    shifted = _score_frozen(shift_pack, coefs, q)
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuf = _score_frozen(
        _pack(ALPHAS_TEST, F_MAX_TEST, x_perm=rng.permutation(shift_pack["x"])),
        coefs,
        q,
    )
    pattern = _pattern(shifted, b0_recall)
    shift_valid = bool(shifted["g_cert"] and shifted["g_safe"])
    if not shifted["g_cert"]:
        locus = "coverage"
    elif not shifted["g_safe"]:
        locus = "safety"
    else:
        locus = None
    status = stage_status()
    status["R7-B0"] = {"go": True, "frozen": True}
    status["R7-B1"] = {
        "shift_valid": shift_valid,
        "pattern": pattern,
        "failure_locus": locus,
        "adapted": False,
    }
    payload = {
        "stage": "R7-B1",
        "scientific_result": True,
        "neural_probe": False,
        "adapted": False,
        "wm_refit": False,
        "q_recalibrated": False,
        "f_max_train": F_MAX,
        "f_max_test": F_MAX_TEST,
        "delta": DELTA,
        "eps": EPS,
        "n_test": len(ALPHAS_TEST),
        "prereg_sha256": _prereg_sha256(),
        "b0_prereg_sha256": b0.get("prereg_sha256"),
        "frozen_q": q,
        "b0_coverage": b0_cov,
        "b0_recall": b0_recall,
        "b0_voi": b0_voi,
        "id_replay": id_replay,
        "shifted": shifted,
        "shuffled": shuf,
        "g_cert": shifted["g_cert"],
        "g_safe": shifted["g_safe"],
        "g_recall": shifted["g_recall"],
        "g_voi": shifted["g_voi"],
        "g_ctrl": bool(shifted["voi"] > shuf["voi"]),
        "pattern": pattern,
        "r7_b1_shift_valid": shift_valid,
        "failure_locus": locus,
        "note": "frozen B0 f_WM and q; only F_max test-time shift; no adaptation",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
