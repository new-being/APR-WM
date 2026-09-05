"""RTWX-O0G5B smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g5b import G5A_SEEDS, K_ANCHOR, RTWXO0G5BConfig, run_rtwx_o0g5b


def test_cli():
    assert build_parser().parse_args(["rtwx-o0g5b"]).command == "rtwx-o0g5b"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0g5b(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWXO0G5BConfig(smoke=True),
        )


def test_frozen():
    assert G5A_SEEDS == (31601, 31602, 31603)
    assert K_ANCHOR == 512


def test_numpy_smoke(tmp_path: Path):
    r = run_rtwx_o0g5b(tmp_path / "g5b", config=RTWXO0G5BConfig(smoke=True))
    assert r["unlocks_o1"] is False
    assert r["unlocks_o0g5r"] is False
    assert r["pattern"] in {
        "target_pipeline_failure",
        "global_canonical_identity_not_representable",
        "global_canonical_identity_supported",
    }
    assert "B1_local" in r and "B2_global" in r
