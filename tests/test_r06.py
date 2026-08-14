import torch

from aprwm_v0.r06 import (
    DISSIPATIVE_OPERATORS,
    V6R06Config,
    constrained_dissipative_map_fit,
    is_dissipative_operator,
)


def test_r06_uses_new_confirmatory_seeds():
    assert set(V6R06Config().seeds).isdisjoint(
        {
            2001, 2011, 2021, 2031, 2041,
            4001, 4011, 4021, 4031, 4041,
            5001, 5011, 5021, 5031, 5041,
            6001, 6011, 6021, 6031, 6041,
        }
    )


def test_only_declared_velocity_families_are_dissipative():
    assert DISSIPATIVE_OPERATORS == (3, 4, 5, 6)
    assert is_dissipative_operator(4)
    assert not is_dissipative_operator(0)


def test_constrained_map_fit_projects_active_coefficient_to_boundary():
    config = V6R06Config()
    tangent = torch.zeros((1, 8, 2))
    velocity = torch.linspace(-1.0, 1.0, 8)
    physical = torch.stack((torch.full_like(velocity, 0.5), velocity), dim=-1)[None]
    target = (velocity * velocity.abs())[None]
    coefficient, projected = constrained_dissipative_map_fit(
        tangent,
        physical,
        target,
        torch.tensor([4]),
        torch.zeros((1, 2)),
        torch.eye(2)[None],
        config,
    )
    assert bool(projected[0])
    assert float(coefficient[0, 2]) == 0.0
