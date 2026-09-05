"""VIS-X0 GT seg+depth interface smoke tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from aprwm_v0.simx_vis_plant import VIS_HOST_PLANT_ID, assert_vis_plant, load_simx_hinge_vis
from aprwm_v0.vis_x0 import VISX0Config, run_vis_x0


def test_vis_x0_refuses_r10_path(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_vis_x0(tmp_path / "runs" / "r10_c0" / "fake")


def test_vis_host_structure():
    model, _data = load_simx_hinge_vis(0.002)
    inv = assert_vis_plant(model)
    assert inv["host_plant_id"] == VIS_HOST_PLANT_ID
    assert inv["nv"] == 1
    assert inv["camera_id"] >= 0
    assert inv["panel_geom_id"] >= 0


def test_vis_x0_short_closure(tmp_path: Path):
    result = run_vis_x0(
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
    # Single-cell smoke: gates evaluated but full 9-cell pass not required.
    assert result["real_physics"] is False
    assert result["real_perception"] is False
    assert result["unlocks_r10_c0"] is False
    assert result["host_plant_id"] == VIS_HOST_PLANT_ID
    assert result["cells"][0]["g_phys"] is True
    assert result["cells"][0]["g_geom"] is True
    cell = tmp_path / "vis_x0" / "seed_9101" / "sine.h5"
    with h5py.File(cell, "r") as handle:
        assert handle.attrs["source"] == "simulator"
        assert handle.attrs["label"] == "vis-x0"
        assert "q_hat" in handle["learner_visible"]
        assert "residual_visual" in handle["learner_visible"]
        assert "q_oracle" in handle["raw_truth"]
