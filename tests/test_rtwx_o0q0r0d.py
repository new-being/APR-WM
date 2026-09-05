"""RTWX-O0Q0R0D smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0 import G0_MAX, TAU
from aprwm_v0.rtwx_o0q0r0c import H_P0, I_RATIO_P0
from aprwm_v0.rtwx_o0q0r0d import (
    EPS_EE,
    EPS_Q,
    PATTERNS,
    SEEDS,
    RTWXO0Q0R0DConfig,
    _pattern,
    run_rtwx_o0q0r0d,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0r0d"]).command == "rtwx-o0q0r0d"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0r0d(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0R0DConfig(smoke=True))


def test_frozen():
    assert SEEDS == (0, 1, 2)
    assert TAU == 0.05
    assert G0_MAX == 0.02
    assert H_P0 == 6
    assert I_RATIO_P0 == 4.0
    assert EPS_Q == 0.05
    assert EPS_EE == 0.03


def test_patterns():
    ok = dict(g0=True, g1=True, g2=True, g3=True, g4=True)
    assert _pattern(**ok) == "native_control_trace_qualified"
    assert _pattern(**{**ok, "g0": False}) == "control_channel_unresolved"
    assert _pattern(**{**ok, "g1": False}) == "capture_incomplete"
    assert _pattern(**{**ok, "g2": False}) == "robot_replay_failure"
    assert _pattern(**{**ok, "g3": False}) == "task_event_replay_failure"
    assert _pattern(**{**ok, "g4": False}) == "precontact_identity_failure"
    assert set(PATTERNS) == {
        "control_channel_unresolved",
        "capture_incomplete",
        "robot_replay_failure",
        "task_event_replay_failure",
        "precontact_identity_failure",
        "native_control_trace_qualified",
    }


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0r0d(tmp_path / "o0q0r0d", config=RTWXO0Q0R0DConfig(smoke=True, backend="numpy"))
    assert r["unlocks_o0q1_prereg"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_hdf5_as_action"] is True
    assert r["header"]["no_yaw_science"] is True
    assert r["header"]["H_P0_frozen"] == 6
    assert r["pattern"] in PATTERNS
    if r["pattern"] != "native_control_trace_qualified":
        assert r["unlocks_o0q0r1_prereg"] is False
    else:
        assert r["unlocks_o0q0r1_prereg"] is True
