import math

import torch

from aprwm_v0.r05p import (
    V6R05PConfig,
    power_violation_metrics,
    structural_power,
)


def test_symmetric_alpha_grid_and_new_seeds():
    config = V6R05PConfig()
    assert config.alpha_magnitudes == (0.15, 0.30, 0.45)
    assert set(config.seeds).isdisjoint(
        {2001, 2011, 2021, 2031, 2041, 4001, 4011, 4021, 4031, 4041}
    )


def test_structural_power_changes_only_with_alpha_sign():
    velocity = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    active = structural_power(0.3, velocity)
    dissipative = structural_power(-0.3, velocity)
    torch.testing.assert_close(active, -dissipative)
    assert bool((active >= 0).all())
    assert bool((dissipative <= 0).all())


def test_power_threshold_is_fixed_by_c0_noise_floor():
    config = V6R05PConfig()
    assert math.isclose(config.power_noise_epsilon, 0.0084)
    maximum, fraction = power_violation_metrics(
        0.3, torch.tensor([0.0, 1.0]), config.power_noise_epsilon
    )
    assert math.isclose(maximum, 0.3, rel_tol=1.0e-6)
    assert fraction == 0.5
