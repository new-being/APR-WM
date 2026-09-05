"""R6-B0: frozen abstention of PX deviations from pi0. Algebraic; no new rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r6_a1 import Z_LOW, pair_key
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R6/R6_B0_PREREG.md"
Z_MIN = Z_LOW


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_stage(path: Path, stage: str, lock: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"R6-B0 locked until {lock} has been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage:
        raise RuntimeError(f"R6-B0 locked: {lock} summary missing")
    return payload


def apply_abstention(a0: str, ax: str, z_switch: float | None) -> tuple[str, str]:
    if a0 == ax:
        return a0, "no_proposal"
    if z_switch is None:
        return a0, "abstain"
    if z_switch >= Z_MIN:
        return ax, "accepted"
    return a0, "abstain"


def run_r6_b0(
    output: str | Path,
    *,
    a1_summary: str | Path = "runs/r6_a1/formal/summary.json",
    d0_summary: str | Path = "runs/r5_d0/formal/summary.json",
) -> dict[str, Any]:
    a1 = _require_stage(Path(a1_summary), "R6-A1", "R6-A1")
    d0 = _require_stage(Path(d0_summary), "R5-D0", "R5-D0")
    if a1.get("z_low") != Z_MIN:
        raise RuntimeError("R6-B0 locked: A1 z_low is not the frozen z_min=1")
    d0_by_lam = {float(row["lam"]): row for row in d0["rows"]}
    px_by_lam = {float(row["lam"]): row for row in a1["table_px"]}
    p0_by_lam = {float(row["lam"]): row for row in a1["table_p0"]}

    rows = []
    n_harm_prop = n_harm_ret = 0
    n_use_prop = n_use_acc = 0
    for lam, d0_row in sorted(d0_by_lam.items()):
        px = px_by_lam[lam]
        p0 = p0_by_lam[lam]
        a0 = p0["hat_a"]
        ax = px["hat_a"]
        star = d0_row["a_star"]
        if a0 != d0_row["a0"] or ax != d0_row["ax"]:
            raise RuntimeError("R6-B0 locked: A1 actions do not match D0")
        z_switch = None
        if a0 != ax:
            z_switch = float(px["pairs"][pair_key(a0, ax)]["z"])
        ab, decision = apply_abstention(a0, ax, z_switch)
        harmful = a0 != ax and ax != star
        useful = a0 != ax and ax == star
        retracted = harmful and ab == a0
        accepted = useful and ab == ax
        n_harm_prop += int(harmful)
        n_harm_ret += int(retracted)
        n_use_prop += int(useful)
        n_use_acc += int(accepted)
        r_b = float(d0_row["r0"] if ab == a0 else d0_row["rx"])
        rows.append(
            {
                "lam": lam,
                "a_star": star,
                "a0": a0,
                "ax": ax,
                "ab": ab,
                "z_switch": z_switch,
                "decision": decision,
                "harmful_proposal": harmful,
                "useful_proposal": useful,
                "harmful_retracted": retracted,
                "useful_accepted": accepted,
                "r0": float(d0_row["r0"]),
                "rx": float(d0_row["rx"]),
                "rb": r_b,
            }
        )

    collapse = bool(all(row["ab"] == row["a0"] for row in rows))
    mean_r0 = float(d0["mean_r0"])
    mean_rx = float(d0["mean_rx"])
    mean_rb = float(np.mean([row["rb"] for row in rows]))
    if collapse:
        mean_rb = mean_r0
    status = stage_status()
    status["R5-D0"] = {"go": False, "selfstress_family_stop": True}
    status["R6-A1"] = {"frozen": True, "b0_licensed": bool(a1.get("b0_licensed"))}
    status["R6-B0"] = {
        "policy_collapse": collapse,
        "realization_branch_stop": True,
    }
    payload = {
        "stage": "R6-B0",
        "scientific_result": True,
        "neural_probe": False,
        "retrained": False,
        "simulator_rerun": False,
        "threshold_swept": False,
        "z_min": Z_MIN,
        "prereg_sha256": _prereg_sha256(),
        "a1_prereg_sha256": a1.get("prereg_sha256"),
        "d0_prereg_sha256": d0.get("prereg_sha256"),
        "b0_policy_collapse": collapse,
        "rows": rows,
        "n_harmful_proposed": n_harm_prop,
        "n_harmful_retracted": n_harm_ret,
        "n_useful_proposed": n_use_prop,
        "n_useful_accepted": n_use_acc,
        "mean_r0": mean_r0,
        "mean_rx": mean_rx,
        "mean_rb": mean_rb,
        "voi_pi_recovered": bool(mean_rb < mean_r0),
        "realization_branch_stop": True,
        "note": (
            "uncertainty repaired harmful realization by abstaining, "
            "but did not recover positive VoI"
            if collapse
            else "frozen abstention applied by D0 regret lookup"
        ),
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
