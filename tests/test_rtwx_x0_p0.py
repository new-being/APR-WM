"""RoboTwin-X0-P0 tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.rtwx_x0_p0 import RTWX0P0Config, run_rtwx_x0_p0


def test_p0_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0_p0(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0P0Config(n_train_ep=1, n_test_ep=1, n_steps=4, latent_epochs=1))
