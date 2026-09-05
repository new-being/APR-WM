"""RTWX-O0Q0 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0q0 import SEEDS, YAW_DEG, RTWXO0Q0Config, _R_y, _pattern, d_Q, run_rtwx_o0q0


def test_cli():
    assert build_parser().parse_args(["rtwx-o0q0"]).command == "rtwx-o0q0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0q0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0Q0Config(smoke=True))


def test_frozen():
    assert SEEDS == (45101, 45102, 45103)
    assert len(YAW_DEG) == 23
    assert 90.0 in YAW_DEG and 0.0 not in YAW_DEG


def test_patterns():
    assert _pattern(g0=False, g1=True, g2=True, g3=True) == "counterfactual_instrument_failure"
    assert _pattern(g0=True, g1=False, g2=True, g3=True) == "causal_symmetry_not_excited"
    assert _pattern(g0=True, g1=True, g2=False, g3=True) == "task_yaw_causally_relevant"
    assert _pattern(g0=True, g1=True, g2=True, g3=True) == "task_yaw_causal_quotient_supported"


def test_dQ_identity():
    z = np.zeros(3)
    q = np.zeros(14)
    y = {"p": z, "n": np.array([0.0, 1.0, 0.0]), "v": z, "nd": z, "q": q, "qd": q, "J": np.array([0.0])}
    assert d_Q(y, y) == 0.0
    assert abs(_R_y(90.0)[0, 2] - 1.0) < 1e-9


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0q0(tmp_path / "o0q0", config=RTWXO0Q0Config(smoke=True, backend="numpy"))
    assert r["unlocks_o1"] is False
    assert r["unlocks_symx3"] is False
    assert r["o0_target_remains_full_TR"] is True
    assert r["header"]["no_rgb"] is True
    assert r["pattern"] in {
        "counterfactual_instrument_failure",
        "causal_symmetry_not_excited",
        "task_yaw_causally_relevant",
        "task_yaw_causal_quotient_supported",
    }
    if r["pattern"] != "task_yaw_causal_quotient_supported":
        assert r["unlocks_o0q1_prereg"] is False
