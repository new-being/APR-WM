import torch

from aprwm_v0.r01 import V6R01Config, _closure_metrics, _pearson


def test_pearson_detects_linear_state_dependence():
    x = torch.linspace(-1.0, 1.0, 32)
    assert abs(_pearson(x, 3.0 * x) - 1.0) < 1.0e-6


def test_c0_closure_metrics_accept_noise_floor_residual():
    config = V6R01Config(force_observation_noise=0.01)
    generator = torch.Generator().manual_seed(9)
    observed = 0.01 * torch.randn((24, 48), generator=generator)
    q = torch.rand((24, 48), generator=generator)
    qvel = 2.0 * torch.rand((24, 48), generator=generator) - 1.0
    data = {
        "clean": torch.zeros_like(observed),
        "observed": observed,
        "features": torch.stack((q, qvel), dim=-1),
    }
    metrics = _closure_metrics(data, config)
    assert metrics["c0_closure_pass"] == 1.0


def test_c0_closure_metrics_reject_position_structure():
    config = V6R01Config(force_observation_noise=0.01)
    q = torch.linspace(-1.0, 1.0, 24 * 48).reshape(24, 48)
    qvel = torch.zeros_like(q)
    observed = 0.05 * q
    data = {
        "clean": observed,
        "observed": observed,
        "features": torch.stack((q, qvel), dim=-1),
    }
    metrics = _closure_metrics(data, config)
    assert metrics["c0_closure_pass"] == 0.0
