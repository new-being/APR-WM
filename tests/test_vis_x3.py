"""VIS-X3 attribution-veto tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.vis_x0 import _local_linear_slope_and_se
from aprwm_v0.vis_x3 import (
    VISX3Config,
    _align_u_to_scores,
    attribution_rates,
    run_vis_x3,
)


def test_vis_x3_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_vis_x3(
            tmp_path / "runs" / "r10_c0" / "fake",
            config=VISX3Config(require_x2_summary=False),
            x2_summary=None,
        )


def test_vis_x3_requires_x2(tmp_path: Path):
    bad = tmp_path / "x2.json"
    bad.write_text(
        '{"vis_x2_passed": false, "false_physics_alarm_evidence": true}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="vis_x2_passed"):
        run_vis_x3(tmp_path / "vis_x3", x2_summary=bad)


def test_slope_se_learner_visible_only():
    t = np.linspace(0, 1, 201)
    q = 0.5 * t + 0.01 * np.sin(20 * t)
    slope, se = _local_linear_slope_and_se(q, dt=t[1] - t[0], half=20)
    assert slope.shape == q.shape
    assert np.all(np.isfinite(se[20:-20]))
    assert float(np.median(se[20:-20])) > 0.0


def test_attribution_rates_shapes():
    scores = np.array([0.1, 0.5, 0.2, 0.8])
    u = np.ones(10)
    u[8:] = 10.0
    rates = attribution_rates(scores, u, theta=0.3, tau_u=2.0, window=7)
    assert "phyclaim_rate" in rates
    assert _align_u_to_scores(u, window=7).size == 4


def test_vis_x3_short_smoke(tmp_path: Path):
    x2_path = tmp_path / "x2.json"
    x2_path.write_text(
        json.dumps(
            {
                "vis_x2_passed": True,
                "false_physics_alarm_evidence": True,
                "alarm_harness": {"theta": 0.3667960699204398},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_vis_x3(
        tmp_path / "vis_x3",
        config=VISX3Config(
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
            g0_abs_tol=1.0,  # smoke: don't require X2 rate match
            g1_d_clean_max=1.0,
            g2_d_gap_min=-1.0,
            g3_fpr_phyclaim_max=1.0,
            g4_tpr_phyclaim_min=0.0,
            require_x2_summary=True,
        ),
        x2_summary=x2_path,
    )
    assert result["unlocks_r10_c0"] is False
    assert "tau_U" in result["harness"]
    assert result["g5_no_oracle_leakage"] is True
    harness = json.loads((tmp_path / "vis_x3" / "alarm_harness.json").read_text())
    assert harness["theta_retune_forbidden"] is True
    assert "FPR_phyclaim_perc" in result["metrics"]
