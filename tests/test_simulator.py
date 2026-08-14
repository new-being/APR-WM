import torch

from aprwm_v0.config import WorldConfig
from aprwm_v0.graph import MASS
from aprwm_v0.simulator import SyntheticWorld


def two_disk_state(material: float) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [-0.10, 0.0, 0.0, 0.0, 0.12, 1.0, 0.4, material],
                [0.10, 0.0, 0.0, 0.0, 0.12, 2.0, 0.4, 0.0],
            ]
        ],
        dtype=torch.float32,
    )


def test_explicit_contact_conserves_pair_momentum_at_rest():
    world = SyntheticWorld(WorldConfig())
    state = two_disk_state(material=0.0)
    output = world.physics_acceleration(state, torch.zeros(1, 2, 2))
    total_force = (output.acceleration * state[..., MASS].unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(total_force, torch.zeros_like(total_force), atol=1e-5, rtol=0)


def test_simple_interaction_is_explained_by_physics():
    world = SyntheticWorld(WorldConfig())
    state = two_disk_state(material=0.0)
    action = torch.zeros(1, 2, 2)
    torch.testing.assert_close(
        world.true_acceleration(state, action).acceleration,
        world.physics_acceleration(state, action).acceleration,
    )


def test_complex_interaction_has_nonzero_residual():
    world = SyntheticWorld(WorldConfig())
    state = two_disk_state(material=1.0)
    action = torch.zeros(1, 2, 2)
    difference = (
        world.true_acceleration(state, action).acceleration
        - world.physics_acceleration(state, action).acceleration
    )
    assert torch.linalg.vector_norm(difference) > 0

