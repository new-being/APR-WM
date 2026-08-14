import torch

from aprwm_v0.v1 import V1Config, V1Data, _decomposition_rows, parameter_posterior


def test_parameter_posterior_recovers_noiseless_linear_physics():
    generator = torch.Generator().manual_seed(4)
    x = torch.rand((64, 32, 2), generator=generator)
    x[..., 0] *= 0.08
    x[..., 1] = 0.4 * x[..., 1] - 0.2
    theta = torch.tensor((32.0, 1.4)).expand(64, -1)
    y = (x * theta.unsqueeze(1)).sum(dim=-1)
    mask = torch.ones((64, 32), dtype=torch.bool)
    mean, covariance, _ = parameter_posterior(x, y, mask, 0.01)
    torch.testing.assert_close(mean.mean(dim=0), theta[0], atol=0.03, rtol=0)
    assert bool((torch.linalg.eigvalsh(covariance) > 0).all())


def test_structural_force_is_zero_below_regime_boundary():
    features = torch.tensor(((0.01, 0.1), (0.034, -0.1), (0.05, 0.0)))
    force = V1Data.structural_force(features)
    assert torch.equal(force[:2], torch.zeros(2))
    assert force[2] > 0


def test_v1_data_separates_structural_class_from_parameters():
    config = V1Config(max_context=32)
    batch = V1Data(torch.device("cpu"), 7, config).sample(2_000)
    assert 0.45 < float(batch["structural"].float().mean()) < 0.55
    adequate = ~batch["structural"]
    assert torch.equal(
        batch["query_structural"][adequate],
        torch.zeros_like(batch["query_structural"][adequate]),
    )


def test_parameter_posterior_uncertainty_shrinks_with_context():
    generator = torch.Generator().manual_seed(12)
    x = torch.rand((1, 32, 2), generator=generator)
    y = (x * torch.tensor((35.0, 1.2))).sum(dim=-1)
    short = torch.arange(32).unsqueeze(0) < 4
    long = torch.ones((1, 32), dtype=torch.bool)
    _, short_covariance, _ = parameter_posterior(x, y, short, 0.08)
    _, long_covariance, _ = parameter_posterior(x, y, long, 0.08)
    assert torch.trace(long_covariance[0]) < torch.trace(short_covariance[0])


def test_error_decomposition_includes_interaction_term():
    conditions = []
    for model_class, parameter_mode, mse in (
        ("adequate", "oracle", 0.0),
        ("adequate", "estimated", 2.0),
        ("inadequate", "oracle", 5.0),
        ("inadequate", "estimated", 4.0),
    ):
        conditions.append(
            {
                "seed": 1,
                "model_class": model_class,
                "parameter_mode": parameter_mode,
                "physics_force_mse": mse,
            }
        )
    row = _decomposition_rows(conditions)[0]
    assert row["parameter_error"] == 2.0
    assert row["structural_error"] == 5.0
    assert row["interaction_term"] == -3.0
