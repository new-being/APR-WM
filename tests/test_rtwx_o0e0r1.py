"""RTWX-O0E0R1 oracle audit tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0e0p0 import RTWXO0E0P0Config, run_rtwx_o0e0p0
from aprwm_v0.rtwx_o0e0r0 import RTWXO0E0R0Config, run_rtwx_o0e0r0
from aprwm_v0.rtwx_o0e0r1 import (
    RTWXO0E0R1Config,
    SEED,
    _axis_diag_pass,
    _patterns,
    run_rtwx_o0e0r1,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0e0r1"]).command == "rtwx-o0e0r1"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0e0r1(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0E0R1Config(p0_cache=str(tmp_path / "p0"), r0_cache=str(tmp_path / "r0")),
        )


def test_patterns():
    p = _patterns(ref_not_invariant=True, ceil_pass=False, formal_pass=False, pgT_pass=True, mask_improves_pos=False)
    assert "reference_offset_not_invariant" in p
    assert "quotient_insufficient_under_oracle" in p
    assert "axis_failure_reference_coupled" in p


def test_axis_diag():
    assert _axis_diag_pass(10.0, 25.0)
    assert not _axis_diag_pass(20.0, 25.0)


def test_smoke_pipeline(tmp_path: Path):
    p0_out = tmp_path / "p0"
    r0_out = tmp_path / "r0"
    r1_out = tmp_path / "r1"
    run_rtwx_o0e0p0(
        p0_out,
        config=replace(RTWXO0E0P0Config(backend="numpy"), n_train=25, n_val=10, n_test=100),
    )
    run_rtwx_o0e0r0(
        r0_out,
        config=RTWXO0E0R0Config(smoke=True, p0_cache=str(p0_out)),
    )
    r1 = run_rtwx_o0e0r1(
        r1_out,
        config=RTWXO0E0R1Config(smoke=True, p0_cache=str(p0_out), r0_cache=str(r0_out)),
    )
    assert "oracle_audit_complete" in r1["patterns"]
    assert "A_reference_offset" in r1
    assert "D_oracle_grid_2x2" in r1
    assert SEED == 39601
