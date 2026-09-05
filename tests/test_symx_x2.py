"""SYM-X2 smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.symx_x2 import SEEDS, SYMX2Config, _revoke_index, run_symx_x2


def test_cli():
    assert build_parser().parse_args(["sym-x2"]).command == "sym-x2"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_symx_x2(tmp_path / "runs" / "r10_c0" / "x", config=SYMX2Config(smoke=True))


def test_frozen_seeds():
    assert SEEDS == (42101, 42102, 42103)


def test_c0fr_not_in_x0_conditions():
    from aprwm_v0.symx_plant import make_bodies
    from aprwm_v0.symx_x0 import CONDITIONS

    assert "C0fr" in make_bodies()
    assert "C0fr" not in CONDITIONS
    assert make_bodies()["C0fr"].mu_aniso == (4.0, 0.0)


def test_revoke_index():
    assert _revoke_index([0.0, 0.0, 0.0], 3) is None
    assert _revoke_index([0.1, 0.1, 0.1], 3) == 3
    assert _revoke_index([0.1, 0.0, 0.1, 0.1, 0.1], 3) == 5


def test_numpy_smoke(tmp_path: Path):
    x0, x1 = Path("runs/symx_x0"), Path("runs/symx_x1")
    if not (x0 / "summary.json").is_file() or not (x1 / "summary.json").is_file():
        pytest.skip("X0/X1 summaries missing")
    r = run_symx_x2(tmp_path / "sx2", config=SYMX2Config(smoke=True, x0_summary=str(x0), x1_summary=str(x1)))
    assert r["unlocks_o1"] is False
    assert r["unlocks_o0g6r"] is False
    assert r["unlocks_symx3"] is False
    assert r["header"]["no_rgb"] is True
    assert r["pattern"] in {
        "instrument_failure",
        "break_undetected",
        "false_revoke",
        "no_recovery",
        "task_world_collapse",
        "gauge_reactivation_supported",
    }
    assert "B1_iner" in r["detection"] and "B3_task" in r["detection"]
