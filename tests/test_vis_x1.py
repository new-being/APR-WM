"""VIS-X1 RGB-D pseudo-residual tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aprwm_v0.vis_x0 import VISX0Config, run_vis_x0
from aprwm_v0.vis_x1 import VISX1Config, panel_mask_from_rgb_depth, run_vis_x1


def test_vis_x1_refuses_r10_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_vis_x1(
            tmp_path / "runs" / "r10_c0" / "fake",
            config=VISX1Config(require_x0_summary=False),
            x0_summary=None,
        )


def test_vis_x1_requires_x0_pass(tmp_path: Path):
    bad = tmp_path / "x0.json"
    bad.write_text('{"vis_x0_passed": false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="vis_x0_passed"):
        run_vis_x1(tmp_path / "vis_x1", x0_summary=bad)


def test_rgb_mask_never_needs_seg():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float64)
    rgb[2:6, 2:6] = (220, 60, 40)
    mask = panel_mask_from_rgb_depth(rgb, depth)
    assert int(mask.sum()) >= 10


def test_vis_x1_short_closure(tmp_path: Path):
    x0 = run_vis_x0(
        tmp_path / "vis_x0",
        config=VISX0Config(
            duration_s=0.30,
            seeds=(9101,),
            trajectories=("sine",),
            slope_half_window=11,
            image_height=120,
            image_width=160,
        ),
    )
    x0_path = tmp_path / "vis_x0" / "summary.json"
    payload = dict(x0)
    payload["vis_x0_passed"] = True
    x0_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    result = run_vis_x1(
        tmp_path / "vis_x1",
        config=VISX1Config(
            duration_s=0.30,
            seeds=(9101,),
            trajectories=("sine",),
            slope_half_window=11,
            image_height=120,
            image_width=160,
        ),
        x0_summary=x0_path,
    )
    assert result["uses_gt_segmentation"] is False
    assert result["unlocks_r10_c0"] is False
    assert result["cells"][0]["g_phys"] is True
    assert result["cells"][0]["g_run"] is True
    assert "E_pseudo" in result["cells"][0]
    cell = tmp_path / "vis_x1" / "seed_9101" / "sine.h5"
    with h5py.File(cell, "r") as handle:
        assert handle.attrs["label"] == "vis-x1"
        assert bool(handle.attrs["uses_gt_segmentation"]) is False
        assert "residual_pseudo" in handle["learner_visible"]
        assert bool(handle["learner_visible"].attrs["excludes_gt_segmentation"])
