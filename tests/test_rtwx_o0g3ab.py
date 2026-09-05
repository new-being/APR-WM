"""RTWX-O0G3A/B smoke tests."""

from __future__ import annotations

from pathlib import Path

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0g3a import RTWXO0G3AConfig, run_rtwx_o0g3a
from aprwm_v0.rtwx_o0g3b import RTWXO0G3BConfig, run_rtwx_o0g3b


def test_cli_has_g3a_g3b():
    assert build_parser().parse_args(["rtwx-o0g3a"]).command == "rtwx-o0g3a"
    assert build_parser().parse_args(["rtwx-o0g3b"]).command == "rtwx-o0g3b"


def test_g3a_smoke(tmp_path: Path):
    r = run_rtwx_o0g3a(tmp_path / "a", config=RTWXO0G3AConfig(g3r_cache="runs/rtwx_o0g3r", smoke=True))
    assert r["pattern"] == "correspondence_availability_characterized"


def test_g3b_smoke(tmp_path: Path):
    r = run_rtwx_o0g3b(tmp_path / "b", config=RTWXO0G3BConfig(g3r_cache="runs/rtwx_o0g3r", smoke=True))
    assert r["pattern"] in {
        "target_pipeline_failure",
        "canonical_correspondence_ambiguous",
        "pixel_coord_not_representable",
        "crop_rgbd_not_representable",
        "correspondence_instrument_closed",
    }
