"""R10-C0: real sensor-chain C0. Locked without a hardware residual log."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


PREREG_PATH = "REPORT/REG/R10/R10_C0_PREREG.md"
SCHEMA_PATH = "REPORT/REG/R10/R10_C0_RESIDUAL_H5_SCHEMA.md"
DEFAULT_LOG = "runs/r10_c0/real/residual.h5"
SCHEMA_ID = "aprwm.r10_c0.residual.v1"


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_r10_c0(
    output: str | Path,
    *,
    residual_log: str | Path = DEFAULT_LOG,
) -> dict[str, Any]:
    del output
    path = Path(residual_log)
    if not path.is_file():
        raise RuntimeError(
            "R10-C0 locked until a real sensor-chain residual log exists "
            f"(missing {path}). source must be hardware|device_log; "
            "Simulator R1-MJ0/RS0 does not unlock this gate."
        )
    raise RuntimeError("R10-C0 log present but evaluator not yet implemented")
