"""RTWX-O0A0 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0a0 import (
    MED_ER_MAX,
    N_YAW,
    P90_ER_MAX,
    RTWXO0A0Config,
    SEED,
    YAW_STEP_DEG,
    _circular_err_deg,
    _pattern,
    _relative_yaw_deg,
    run_rtwx_o0a0,
)
from aprwm_v0.rtwx_o0q0 import _R_to_quat, _R_y
from aprwm_v0.rtwx_o0 import _quat_fix


def test_cli():
    assert build_parser().parse_args(["rtwx-o0a0"]).command == "rtwx-o0a0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0a0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0A0Config(smoke=True))


def test_frozen():
    assert SEED == 35601
    assert N_YAW == 24
    assert YAW_STEP_DEG == 15.0
    assert MED_ER_MAX == 15.0
    assert P90_ER_MAX == 30.0


def test_patterns():
    assert _pattern(False, True, False) == "appearance_instrument_failure"
    assert _pattern(True, False, False) == "appearance_yaw_generalization_failure"
    assert _pattern(True, True, False) == "appearance_yaw_supported"
    assert _pattern(True, True, True) == "geometry_yaw_supported_unexpectedly"
    assert _pattern(True, False, True) == "geometry_yaw_supported_unexpectedly"


def test_circular_and_relative_yaw():
    assert _circular_err_deg(np.array([350.0]), np.array([10.0]))[0] == pytest.approx(20.0)
    q0 = _quat_fix(np.array([1.0, 0.0, 0.0, 0.0]))[0]
    q = _R_to_quat(_R_y(90.0))
    assert abs(_relative_yaw_deg(q0, q) - 90.0) < 1e-4


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0a0(tmp_path / "o0a0", config=RTWXO0A0Config(smoke=True, backend="numpy"))
    assert r["header"]["encoder"] == "coordconv_spatial_flatten_no_GAP"
    assert r["unlocks_o1"] is False
    assert r["pattern"] in {
        "appearance_instrument_failure",
        "appearance_yaw_generalization_failure",
        "appearance_yaw_supported",
        "geometry_yaw_supported_unexpectedly",
    }
    if r["pattern"] == "appearance_yaw_supported":
        assert r["unlocks_o0a1_prereg"] is True
        assert r["unlocks_o0t0_prereg"] is False
    elif r["pattern"] == "appearance_yaw_generalization_failure":
        assert r["unlocks_o0t0_prereg"] is True
        assert r["unlocks_o0a1_prereg"] is False
    else:
        assert r["unlocks_o0a1_prereg"] is False
        assert r["unlocks_o0t0_prereg"] is False
