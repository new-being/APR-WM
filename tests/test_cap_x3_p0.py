"""CAP-X3-P0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cap_x3_p0 import CAPX3P0Config, run_cap_x3_p0
from aprwm_v0.cap_x3_theta import D_ACTIVE, active_to_theta, nominal_active, theta_to_active
from aprwm_v0.capx_arm3_plant import sample_scene_theta


def test_active_roundtrip():
    rng = np.random.default_rng(1)
    th = sample_scene_theta(rng)
    a = theta_to_active(th)
    assert a.shape == (D_ACTIVE,)
    back = active_to_theta(a)
    assert np.allclose(theta_to_active(back), a)
    assert np.allclose(back.frictionloss, 0.0)
    assert nominal_active().shape == (D_ACTIVE,)


def test_p0_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_cap_x3_p0(tmp_path / "runs" / "r10_c0" / "x", config=CAPX3P0Config(n_meta_scenes=1, n_probe_scenes=1, n_cal=1, n_query=1, n_meta_traj=1, latent_epochs=1, latent_adapt_steps=2, physics_maxiter=2))
