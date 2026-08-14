import torch

from aprwm_v0.v3 import (
    V3Config,
    _classify_revision,
    _ridge_fit,
    structural_force,
    tangent_decomposition,
)


def test_tangent_projection_separates_parameter_and_structural_directions():
    generator = torch.Generator().manual_seed(9)
    jacobian = torch.randn((8, 12, 2), generator=generator)
    parameter_delta = torch.tensor((0.7, -0.2)).expand(8, -1)
    parameter_residual = torch.einsum("bni,bi->bn", jacobian, parameter_delta)
    _, orthogonal, score = tangent_decomposition(jacobian, parameter_residual)
    assert float(orthogonal.abs().max()) < 1.0e-4
    assert float(score.max()) < 1.0e-6


def test_cubic_discrepancy_is_not_parameter_explainable_over_a_window():
    config = V3Config(episodes=4)
    generator = torch.Generator().manual_seed(3)
    x = 0.015 + 0.07 * torch.rand((4, 16), generator=generator)
    v = -0.2 + 0.4 * torch.rand((4, 16), generator=generator)
    features = torch.stack((x, v), dim=-1)
    residual = structural_force(features, torch.ones(4, dtype=torch.long), config)
    _, orthogonal, score = tangent_decomposition(features, residual)
    assert float(orthogonal.square().mean().sqrt()) > config.observation_noise
    assert float(score.mean()) > 0.1


def test_revision_accepts_cubic_but_rejects_unseen_family():
    config = V3Config(episodes=2, revision_improvement_threshold=1.0e-5)
    x = torch.linspace(0.015, 0.085, 20).repeat(2, 1)
    v = torch.linspace(-0.2, 0.2, 20).repeat(2, 1)
    features = torch.stack((x, v), dim=-1)
    theta = torch.tensor(((35.0, 1.2), (35.0, 1.2)))
    classes = torch.tensor((1, 2))
    target = (features * theta.unsqueeze(1)).sum(dim=-1) + structural_force(
        features, classes, config
    )
    state, _, _ = _classify_revision(
        torch.ones(2, dtype=torch.bool),
        features[:, :16],
        target[:, :16],
        features[:, 16:],
        target[:, 16:],
        config,
    )
    assert state.tolist() == [1, 2]


def test_ridge_fit_recovers_linear_parameters():
    generator = torch.Generator().manual_seed(2)
    design = torch.randn((6, 20, 2), generator=generator)
    theta = torch.tensor((2.5, -0.7)).expand(6, -1)
    target = torch.einsum("bni,bi->bn", design, theta)
    estimate = _ridge_fit(design, target, 1.0e-7)
    torch.testing.assert_close(estimate, theta, atol=1.0e-5, rtol=1.0e-5)
