"""RTWX-O0Q0R0C smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0 import EY, G0_MAX, TAU
from aprwm_v0.rtwx_o0q0r0c import (
    H_P0,
    I_RATIO_P0,
    N_P2_MIN,
    N_P3_MIN,
    P0_W_C,
    PATTERNS,
    SEEDS,
    RTWXO0Q0R0CConfig,
    _pattern,
    run_rtwx_o0q0r0c,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0r0c"]).command == "rtwx-o0q0r0c"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0r0c(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0R0CConfig(smoke=True))


def test_frozen():
    assert SEEDS == (0, 1, 2)
    assert TAU == 0.05
    assert G0_MAX == 0.02
    assert H_P0 == 6
    assert I_RATIO_P0 == 4.0
    assert N_P2_MIN == 3
    assert N_P3_MIN == 2
    w = np.asarray(P0_W_C, dtype=np.float64)
    assert float(np.linalg.norm(np.cross(w, EY))) > 0.1


def test_patterns():
    ok = dict(g0=True, g1=True, g2=True, g3=True, g4=True, g5=True)
    assert _pattern(**ok) == "counterfactual_instrument_qualified"
    assert _pattern(**{**ok, "g0": False}) == "identity_regression_failure"
    assert _pattern(**{**ok, "g1": False}) == "native_demo_action_replay_failure"
    assert _pattern(**{**ok, "g2": False}) == "precontact_event_repro_failure"
    assert _pattern(**{**ok, "g3": False}) == "freeflight_excitation_failure"
    assert _pattern(**{**ok, "g4": False}) == "contact_excitation_failure"
    assert _pattern(**{**ok, "g5": False}) == "contact_excitation_failure"
    assert set(PATTERNS) == {
        "identity_regression_failure",
        "native_demo_action_replay_failure",
        "precontact_event_repro_failure",
        "freeflight_excitation_failure",
        "contact_excitation_failure",
        "counterfactual_instrument_qualified",
    }


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0r0c(tmp_path / "o0q0r0c", config=RTWXO0Q0R0CConfig(smoke=True, backend="numpy"))
    assert r["unlocks_o0q1_prereg"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_ik"] is True
    assert r["header"]["no_restore_contact"] is True
    assert r["header"]["no_yaw_science"] is True
    assert r["header"]["H_P0"] == 6
    assert r["header"]["I_ratio_p0"] == 4.0
    assert r["pattern"] in PATTERNS
    if r["pattern"] != "counterfactual_instrument_qualified":
        assert r["unlocks_o0q0r1_prereg"] is False
    else:
        assert r["unlocks_o0q0r1_prereg"] is True
