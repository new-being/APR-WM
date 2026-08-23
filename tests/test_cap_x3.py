"""CAP-X3 formal tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cap_x3 import CAPX3Config, k90_of_curve, run_cap_x3


def test_k90_curve():
    assert k90_of_curve(1.0, {1: 0.9, 2: 0.5, 4: 0.2, 8: 0.15, 16: 0.1}) == 8
    assert k90_of_curve(0.1, {1: 0.1, 2: 0.1, 4: 0.1, 8: 0.1, 16: 0.1}) == 32


def test_cap_x3_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_cap_x3(tmp_path / "runs" / "r10_c0" / "x", config=CAPX3Config(n_test=1, n_train=1, n_val=1))


def test_cap_x3_requires_p0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="P0"):
        run_cap_x3(tmp_path / "cap_x3", config=CAPX3Config(p0_dir=str(tmp_path / "missing"), n_test=1, n_train=1, n_val=1))
