"""VIS-EXT0 Door visual bridge tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.vis_ext0 import (
    VISEXT0Config,
    apply_vision_rgb_depth,
    run_vis_ext0,
)


def test_vis_ext0_refuses_r10_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_vis_ext0(
            tmp_path / "runs" / "r10_c0" / "fake",
            config=VISEXT0Config(require_x3_summary=False),
            x3_summary=None,
        )


def test_vis_ext0_requires_x3_pass(tmp_path: Path):
    bad = tmp_path / "x3.json"
    bad.write_text('{"vis_x3_passed": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="vis_x3_passed"):
        run_vis_ext0(tmp_path / "vis_ext0", x3_summary=bad)


def test_vision_factors_change_rgb():
    rng = np.random.default_rng(0)
    rgb = np.full((32, 32, 3), 180, dtype=np.uint8)
    depth = np.full((32, 32), 0.9, dtype=np.float64)
    tex, _ = apply_vision_rgb_depth(rgb, depth, vision="texture_light", rng=rng)
    assert tex.mean() < rgb.mean()
    occ, d2 = apply_vision_rgb_depth(rgb, depth, vision="occlusion", rng=rng)
    assert np.any(occ == 0)
    assert np.any(d2 > 100.0)


@pytest.mark.slow
def test_vis_ext0_short_smoke(tmp_path: Path):
    x3 = tmp_path / "x3.json"
    x3.write_text(
        json.dumps({"vis_x3_passed": True, "outcome_pattern": "attribution_success"})
        + "\n",
        encoding="utf-8",
    )
    result = run_vis_ext0(
        tmp_path / "vis_ext0",
        config=VISEXT0Config(
            duration_s=0.30,
            slope_half_window=7,
            alarm_window=20,
            cal_seed=9101,
            cal_trajectory="sine",
            test_seeds=(9111,),
            test_trajectories=("sine",),
            require_x3_summary=True,
        ),
        x3_summary=x3,
    )
    assert result["unlocks_r10_c0"] is False
    assert "theta" in result["alarm_harness"]
    assert "tau_U" in result["alarm_harness"]
    assert result["alarm_harness"]["copied_from_hinge"] is False
    assert "G0_physics_closure" in result["gates"]
    assert (tmp_path / "vis_ext0" / "summary.json").is_file()
