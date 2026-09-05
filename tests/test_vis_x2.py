"""VIS-X2 false physics-diagnosis tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.vis_x1 import VISX1Config, run_vis_x1
from aprwm_v0.vis_x0 import VISX0Config, run_vis_x0
from aprwm_v0.vis_x2 import (
    VISX2Config,
    degrade_rgb_depth,
    run_vis_x2,
    window_scores,
)


def test_vis_x2_refuses_r10_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_vis_x2(
            tmp_path / "runs" / "r10_c0" / "fake",
            config=VISX2Config(require_x1_summary=False),
            x1_summary=None,
        )


def test_vis_x2_requires_x1_pass(tmp_path: Path):
    bad = tmp_path / "x1.json"
    bad.write_text('{"vis_x1_passed": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="vis_x1_passed"):
        run_vis_x2(tmp_path / "vis_x2", x1_summary=bad)


def test_degrade_and_window_scores():
    rng = np.random.default_rng(0)
    rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
    depth = np.full((32, 32), 1.2, dtype=np.float64)
    d_rgb, d_depth = degrade_rgb_depth(rgb, depth, rng=rng)
    assert d_rgb.dtype == np.uint8
    assert d_rgb.mean() < rgb.mean()
    assert np.any(d_depth > 100.0)
    r = np.sin(np.linspace(0, 6, 200))
    tau = np.ones(200)
    s = window_scores(r, tau, window=20, eps=1e-8)
    assert s.size == 181
    assert np.all(np.isfinite(s))


def test_vis_x2_short_smoke(tmp_path: Path):
    # Minimal X0→X1 stubs with forced pass flags.
    x0 = run_vis_x0(
        tmp_path / "vis_x0",
        config=VISX0Config(
            duration_s=0.20,
            seeds=(9101,),
            trajectories=("sine",),
            slope_half_window=7,
            image_height=80,
            image_width=120,
        ),
    )
    x0_path = tmp_path / "vis_x0" / "summary.json"
    payload = dict(x0)
    payload["vis_x0_passed"] = True
    x0_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    x1 = run_vis_x1(
        tmp_path / "vis_x1",
        config=VISX1Config(
            duration_s=0.20,
            seeds=(9101,),
            trajectories=("sine",),
            slope_half_window=7,
            image_height=80,
            image_width=120,
        ),
        x0_summary=x0_path,
    )
    x1_path = tmp_path / "vis_x1" / "summary.json"
    payload1 = dict(x1)
    payload1["vis_x1_passed"] = True
    x1_path.write_text(json.dumps(payload1, indent=2) + "\n", encoding="utf-8")

    result = run_vis_x2(
        tmp_path / "vis_x2",
        config=VISX2Config(
            duration_s=0.40,
            slope_half_window=7,
            image_height=80,
            image_width=120,
            alarm_window=20,
            cal_seed=9101,
            cal_trajectory="sine",
            test_seeds=(9111,),
            test_trajectories=("sine",),
            mismatch_dampings=(0.05,),
            tpr_margin_over_fpr=0.0,
            fpr_perc_margin_over_clean=0.0,
        ),
        x1_summary=x1_path,
    )
    assert result["unlocks_r10_c0"] is False
    assert "FPR_perc" in result["metrics"]
    assert "theta" in result["alarm_harness"]
    assert result["alarm_harness"]["perception_uncertainty_veto"] is False
    assert result["g_phys_clean"] is True
    assert (tmp_path / "vis_x2" / "alarm_harness.json").is_file()
