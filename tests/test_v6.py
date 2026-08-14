import torch

from aprwm_v0.v6 import (
    V6Config,
    _sequential_decision,
    fixed_acceptance,
    noise_calibrated_excess_mse,
)


def test_noise_calibration_removes_known_observation_variance():
    squared_error = torch.full((3, 8), 0.01)
    excess = noise_calibrated_excess_mse(squared_error, 0.01)
    torch.testing.assert_close(excess, torch.zeros(3))


def test_calibration_prevents_noise_floor_from_forcing_rejection():
    config = V6Config(acceptance_rmse=0.095, acceptance_margin=0.0)
    incumbent = torch.full((1, 8), 0.04)
    candidate = torch.full((1, 8), 0.01)
    complexity = torch.zeros(1)
    raw = fixed_acceptance(
        incumbent, incumbent, candidate, complexity, 0.01, config, calibrated=False
    )
    calibrated = fixed_acceptance(
        incumbent, incumbent, candidate, complexity, 0.01, config, calibrated=True
    )
    assert not bool(raw[0])
    assert bool(calibrated[0])


def test_sequential_bf_accepts_better_and_rejects_worse_candidate():
    config = V6Config(
        validation_max=16,
        validation_step=4,
        validation_min=4,
        acceptance_rmse=1.0,
        bf_accept=2.0,
        bf_reject=-2.0,
        bf_complexity_log_prior=0.0,
    )
    incumbent = torch.stack((torch.full((16,), 0.25), torch.full((16,), 0.01)))
    candidate = torch.stack((torch.full((16,), 0.01), torch.full((16,), 0.25)))
    accepted, rejected, deferred, samples = _sequential_decision(
        incumbent,
        incumbent,
        candidate,
        torch.zeros(2),
        0.01,
        config,
        method="bf",
    )
    assert bool(accepted[0]) and bool(rejected[1])
    assert not bool(deferred.any())
    assert bool((samples <= 16).all())


def test_sequential_lcb_rejects_model_without_positive_gain():
    config = V6Config(
        validation_max=16,
        validation_step=4,
        validation_min=4,
        acceptance_rmse=1.0,
    )
    incumbent = torch.full((1, 16), 0.02)
    candidate = incumbent.clone()
    accepted, rejected, deferred, samples = _sequential_decision(
        incumbent,
        incumbent,
        candidate,
        torch.zeros(1),
        0.01,
        config,
        method="lcb",
    )
    assert not bool(accepted[0])
    assert bool(rejected[0])
    assert not bool(deferred[0])
    assert samples[0] == 4
