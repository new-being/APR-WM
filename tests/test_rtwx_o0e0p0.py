"""RTWX-O0E0P0 instrument tests. No B2 science."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0 import e_axis_deg
from aprwm_v0.rtwx_o0e0p0 import (
    EXCITE_RHO,
    N_TEST,
    P_ANY_AGG,
    SEED,
    TILT_DEG,
    YAW_OCTANTS_MIN,
    RTWXO0E0P0Config,
    _pattern,
    n_from_tilt,
    run_rtwx_o0e0p0,
    sample_pose,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0p0"]).command == "rtwx-o0e0p0"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0p0(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0E0P0Config(smoke=True, backend="numpy"))


def test_frozen():
    assert SEED == 37601
    assert TILT_DEG == (0.0, 10.0, 20.0, 35.0, 50.0)
    assert P_ANY_AGG == 0.98
    assert EXCITE_RHO == 0.30
    assert YAW_OCTANTS_MIN == 6
    assert N_TEST == 200


def test_patterns():
    assert _pattern(g0a=False, g0b=True, g0c=True) == "observation_support_failure"
    assert _pattern(g0a=True, g0b=False, g0c=True) == "axis_excitation_failure"
    assert _pattern(g0a=True, g0b=True, g0c=False) == "yaw_nuisance_coverage_failure"
    assert _pattern(g0a=True, g0b=True, g0c=True) == "controlled_pose_observation_qualified"


def test_tilt_geometry():
    n0 = n_from_tilt(0.0, 0.0)
    assert e_axis_deg(n0, np.array([0.0, 0.0, 1.0])) < 1e-6
    n50 = n_from_tilt(50.0, 0.0)
    assert abs(e_axis_deg(n50, np.array([0.0, 0.0, 1.0])) - 50.0) < 1e-4


def test_excite_fraction_design():
    assert sum(1 for b in TILT_DEG if b > 15.0) / len(TILT_DEG) == pytest.approx(0.6)


def test_yaw_randomized():
    rng = np.random.default_rng(0)
    gammas = [float(sample_pose(rng, 20.0)["gamma_rad"]) for _ in range(40)]
    assert max(gammas) - min(gammas) > 1.0


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0e0p0(tmp_path / "p0", config=RTWXO0E0P0Config(smoke=True, backend="numpy"))
    assert r["B2_not_run"] is True
    assert r["b2_untested"] is True
    assert r["unlocks_o1"] is False
    assert r["header"]["not_natural_task_claim"] is True
    assert r["pattern"] in {
        "observation_support_failure",
        "axis_excitation_failure",
        "yaw_nuisance_coverage_failure",
        "controlled_pose_observation_qualified",
    }
    z = np.load(tmp_path / "p0" / "cache" / "test" / "obs.npz")
    assert "p_gt" not in z.files and "n_gt" not in z.files
    if r["pattern"] == "controlled_pose_observation_qualified":
        assert r["unlocks_o0e0_science_on_p0_cache"] is True
        assert r["G0b"]["r_excite"] >= 0.30
