"""RTWX-O0REL0A tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0rel0a import (
    REL0_FORMAL_PATTERN,
    RTWXO0REL0AConfig,
    diagnostic_c_overlap_curve,
    run_rtwx_o0rel0a,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0rel0a"]).command == "rtwx-o0rel0a"


def test_overlap_bins_supported():
    pooled = []
    for rho, e in [(0.05, 22.0), (0.06, 18.0), (0.4, 6.0), (0.5, 4.0)]:
        pooled.append({"rho_overlap": rho, "e_delta_n_deg": e, "regime": "c0"})
    out = diagnostic_c_overlap_curve(pooled)
    assert out["overlap_support_hypothesis_supported"]


def test_formal_run(tmp_path: Path):
    rel0 = Path("/root/APR-WM/runs/rtwx_o0rel0")
    if not (rel0 / "summary.json").is_file():
        pytest.skip("REL0 formal run not available")
    out = run_rtwx_o0rel0a(
        tmp_path / "rel0a",
        config=RTWXO0REL0AConfig(rel0_run=str(rel0)),
    )
    assert out["rel0_formal_pattern"] == REL0_FORMAL_PATTERN
    assert out["rel0_formal_pattern_unchanged"] is True
    assert "diagnostic_A_gtmask_icp_c0" in out
