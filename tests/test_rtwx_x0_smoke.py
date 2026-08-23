"""RoboTwin-X0 official smoke tests (no SAPIEN required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.rtwx_x0_smoke import RTWX0SmokeConfig, run_rtwx_x0_smoke


def test_smoke_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0_smoke(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0SmokeConfig())
