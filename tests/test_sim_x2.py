"""SIM-X2 action-conditioned information channel tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.sim_x2 import (
    SIMX2Config,
    action_torque_sequence,
    mi_binary_v_given_mixture,
    run_sim_x2,
)


def test_analytic_mi_zero_when_no_separation():
    mi = mi_binary_v_given_mixture(
        mean_valid=0.0, mean_inv_a=0.0, mean_inv_b=0.0, sigma=0.01
    )
    assert mi < 1.0e-6


def test_analytic_mi_positive_when_separated():
    mi = mi_binary_v_given_mixture(
        mean_valid=0.0, mean_inv_a=-0.05, mean_inv_b=0.05, sigma=0.01
    )
    assert mi > 0.5


def test_info_and_low_f_same_waveform():
    n = 1000
    dt = 0.002
    a = action_torque_sequence("info", n_steps=n, dt=dt)
    b = action_torque_sequence("low_f", n_steps=n, dt=dt)
    assert (a == b).all()
    cons = action_torque_sequence("cons", n_steps=n, dt=dt)
    high = action_torque_sequence("high_f", n_steps=n, dt=dt)
    assert abs(float((a**2).mean()) - float((high**2).mean())) < 1.0e-12
    assert float((cons**2).mean()) < float((a**2).mean())


def test_sim_x2_locked_without_x1(tmp_path: Path):
    with pytest.raises(RuntimeError, match="SIM-X1 summary"):
        run_sim_x2(tmp_path / "x2", x1_summary=tmp_path / "missing.json")


def test_sim_x2_forbids_sigma_retune(tmp_path: Path):
    x1 = tmp_path / "x1.json"
    x1.write_text(
        '{"sim_x1_passed": true, "host_plant_id": "simx_hinge.v1"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="sigma_obs"):
        run_sim_x2(
            tmp_path / "x2",
            config=SIMX2Config(sigma_obs=0.02, duration_s=3.0, steady_t0_s=1.0),
            x1_summary=x1,
        )


def test_sim_x2_short_channel(tmp_path: Path):
    x1 = tmp_path / "x1.json"
    x1.write_text(
        '{"sim_x1_passed": true, "host_plant_id": "simx_hinge.v1"}',
        encoding="utf-8",
    )
    result = run_sim_x2(
        tmp_path / "sim_x2",
        config=SIMX2Config(duration_s=4.0, steady_t0_s=1.0, seeds=(9101,)),
        x1_summary=x1,
    )
    assert result["unlocks_r10_c0"] is False
    assert result["real_physics"] is False
    assert result["sigma_obs_kind"] == "synthetic_observation_resolution_surrogate"
    assert result["g0_x1_carryover"] is True
    assert result["g1_action_isolation"] is True
    assert result["I_bits_per_sample"]["info"] > result["I_bits_per_sample"]["cons"]
    assert result["E_v"]["low_f"] > result["E_v"]["high_f"]
