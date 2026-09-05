"""SYM-X0 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.symx_plant import axis_angle_R, d_state, make_bodies
from aprwm_v0.symx_x0 import SEEDS, SYMX0Config, in_Gstar, run_symx_x0


def test_cli():
    assert build_parser().parse_args(["sym-x0"]).command == "sym-x0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_symx_x0(tmp_path / "runs" / "r10_c0" / "x", config=SYMX0Config(smoke=True))


def test_frozen():
    assert SEEDS == (40101, 40102, 40103)


def test_gstar():
    I = np.eye(3)
    assert in_Gstar(I, "C1")
    assert in_Gstar(axis_angle_R(np.array([0.0, 1.0, 0.0]), 90.0), "C0")
    assert in_Gstar(axis_angle_R(np.array([1.0, 0.0, 0.0]), 180.0), "C0")
    assert not in_Gstar(axis_angle_R(np.array([1.0, 0.0, 0.0]), 90.0), "C0")
    assert not in_Gstar(axis_angle_R(np.array([0.0, 1.0, 0.0]), 90.0), "C4")
    assert in_Gstar(axis_angle_R(np.array([0.0, 1.0, 0.0]), 180.0), "C4")
    assert in_Gstar(axis_angle_R(np.array([1.0, 0.0, 0.0]), 90.0), "C3")
    assert not in_Gstar(axis_angle_R(np.array([0.0, 1.0, 0.0]), 90.0), "C1")


def test_d_ignores_logo():
    b = make_bodies()
    assert b["C5"].visual_logo is not None
    s = {"p": np.zeros(3), "R": np.eye(3), "v": np.zeros(3), "w": np.zeros(3)}
    assert d_state(s, s) == 0.0


def test_numpy_smoke(tmp_path: Path):
    r = run_symx_x0(tmp_path / "sx0", config=SYMX0Config(smoke=True))
    assert r["unlocks_o1"] is False
    assert r["unlocks_symx3"] is False
    assert r["header"]["no_rgb_in_d"] is True
    assert r["pattern"] in {
        "instrument_failure",
        "shape_symmetry_confound",
        "appearance_confound",
        "causal_symmetry_supported",
        "symmetry_discovery_insufficient",
    }
    c0 = r["per_condition"]["C0"]
    assert c0["D_H_Ry90"] < c0["D_H_Rx90"] or not np.isfinite(c0["D_H_Rx90"])
