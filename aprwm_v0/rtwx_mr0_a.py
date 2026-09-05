"""RTWX-MR0-A: action-only resampling sweep. Blocked until MR0-P0 qualifies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluation.action_resampling_metrics import ACTION_STRIDES
from .rtwx_mr0 import MR0_TASKS, N_EP, paired_episode_seeds
from .rtwx_o0hyb0 import _resolve_apr
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/wam/RTWX0MR0A_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_mr0_a.action_resampling.v1"
CELLS = ((1, 1), (1, 2), (1, 4), (1, 8))


@dataclass(frozen=True)
class RTWXMR0AConfig:
    output: str = "runs/rtwx_mr0_a"
    p0_run: str = "runs/rtwx_mr0_p0"
    seed: int = 49602
    smoke: bool = False


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-MR0-A must not write there")


def load_p0_summary(p0_run: str) -> dict[str, Any]:
    p = _resolve_apr(p0_run) / "summary.json"
    if not p.is_file():
        return {"pattern": "missing", "unlocks_mr0_a": False}
    import json

    return json.loads(p.read_text())


def run_rtwx_mr0_a(output: str | Path, config: RTWXMR0AConfig | None = None) -> dict[str, Any]:
    cfg = config or RTWXMR0AConfig(output=str(output))
    root = _resolve_apr(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    p0 = load_p0_summary(cfg.p0_run)
    if p0.get("pattern") != "action_chunk_qualified":
        summary = {
            "pattern": "mr0_a_blocked_unqualified_policy",
            "scientific_result": False,
            "p0_pattern": p0.get("pattern"),
            "cells": [list(c) for c in CELLS],
            "action_strides": list(ACTION_STRIDES),
            "tasks": list(MR0_TASKS),
            "episodes_per_task": N_EP,
            "paired_seeds": paired_episode_seeds(N_EP),
            "note": "MR0-A requires P0 action_chunk_qualified. Ks is frozen at 1; S-sweep is not in this cell.",
        }
        _write_json(root / "header.json", {"schema_id": SCHEMA_ID, "prereg": PREREG_PATH})
        _write_json(root / "summary.json", summary)
        _write_json(root / "run.json", {"config": asdict(cfg), "summary": summary, "p0": p0})
        print("[rtwx-mr0-a] pattern=mr0_a_blocked_unqualified_policy", flush=True)
        return summary
    raise RuntimeError("P0 qualified but MR0-A task loop is not wired")
