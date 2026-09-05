"""RTWX-O0Q0R0 smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0r0 import (
    G0_MAX,
    SEEDS,
    TAU,
    RTWXO0Q0R0Config,
    _pattern,
    run_rtwx_o0q0r0,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0r0"]).command == "rtwx-o0q0r0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0r0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0R0Config(smoke=True))


def test_frozen():
    assert SEEDS == (0, 1, 2)
    assert TAU == 0.05
    assert G0_MAX == 0.02


def test_patterns():
    assert _pattern(g0=False, g_native=True, g1=True) == "counterfactual_restore_failure"
    assert _pattern(g0=True, g_native=False, g1=True) == "native_contact_snapshot_failure"
    assert _pattern(g0=True, g_native=True, g1=False) == "regime_excitation_failure"
    assert _pattern(g0=True, g_native=True, g1=True) == "counterfactual_instrument_qualified"


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0r0(tmp_path / "o0q0r0", config=RTWXO0Q0R0Config(smoke=True, backend="numpy"))
    assert r["unlocks_o0q1_prereg"] is False
    assert r["unlocks_o1"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_rgb"] is True
    assert r["header"]["no_teleport_into_contact"] is True
    assert r["header"]["no_yaw_science"] is True
    assert r["pattern"] in {
        "counterfactual_restore_failure",
        "native_contact_snapshot_failure",
        "regime_excitation_failure",
        "counterfactual_instrument_qualified",
    }
    if r["pattern"] != "counterfactual_instrument_qualified":
        assert r["unlocks_o0q0r1_prereg"] is False
    else:
        assert r["unlocks_o0q0r1_prereg"] is True
