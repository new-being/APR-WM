"""Unit tests for R10-C0 real-C0 lock."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r10_c0 import DEFAULT_LOG, run_r10_c0


def test_r10_c0_locked_without_real_log(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until a real sensor-chain"):
        run_r10_c0(tmp_path, residual_log=tmp_path / "missing.h5")
    assert DEFAULT_LOG.endswith("residual.h5")
    src = Path("aprwm_v0/r10_c0.py").read_text(encoding="utf-8")
    assert "Simulator R1-MJ0/RS0 does not unlock" in src
