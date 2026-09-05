"""RTWX-O0Q0R0B smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0r0b import (
    DELTA_T,
    DT_A,
    G0_MAX,
    SEEDS,
    TAU,
    RTWXO0Q0R0BConfig,
    _pattern,
    h_p0,
    run_rtwx_o0q0r0b,
    t_hit,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0r0b"]).command == "rtwx-o0q0r0b"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0r0b(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0R0BConfig(smoke=True))


def test_frozen():
    assert SEEDS == (0, 1, 2)
    assert TAU == 0.05
    assert G0_MAX == 0.02
    assert DELTA_T == 1
    assert abs(DT_A - 0.02) < 1e-12


def test_ballistic_H():
    t = t_hit(0.12, 0.0)
    assert abs(t - np.sqrt(2 * 0.12 / 9.81)) < 1e-9
    hp = h_p0(0.12, 0.0)
    assert hp >= 1
    assert hp == int(np.floor((t - 0.03) / 0.02))


def test_patterns():
    assert _pattern(p0_ok=False, branch_ok=True, g3=True) == "freeflight_horizon_failure"
    assert _pattern(p0_ok=True, branch_ok=False, g3=True) == "precontact_branch_failure"
    assert _pattern(p0_ok=True, branch_ok=True, g3=False) == "regime_excitation_failure"
    assert _pattern(p0_ok=True, branch_ok=True, g3=True) == "counterfactual_instrument_qualified"


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0r0b(tmp_path / "o0q0r0b", config=RTWXO0Q0R0BConfig(smoke=True, backend="numpy"))
    assert r["unlocks_o0q1_prereg"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_restore_contact"] is True
    assert r["header"]["no_yaw_science"] is True
    assert r["pattern"] in {
        "freeflight_horizon_failure",
        "precontact_branch_failure",
        "regime_excitation_failure",
        "counterfactual_instrument_qualified",
    }
    if r["pattern"] != "counterfactual_instrument_qualified":
        assert r["unlocks_o0q0r1_prereg"] is False
    else:
        assert r["unlocks_o0q0r1_prereg"] is True
