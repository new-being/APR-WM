import torch

from aprwm_v0.v4 import (
    COUPLED_OPERATOR,
    CUBIC_OPERATOR,
    OPERATOR_NAMES,
    V4Config,
    _ensure_oracle_proposal,
    operator_library,
    orthogonal_operator_gain,
)


def test_operator_library_has_declared_shape_and_finite_values():
    features = torch.randn((5, 11, 2)) * 0.1
    operators = operator_library(features)
    assert operators.shape == (5, 11, len(OPERATOR_NAMES))
    assert torch.isfinite(operators).all()


def test_orthogonal_gain_ranks_cubic_for_cubic_residual():
    generator = torch.Generator().manual_seed(8)
    x = 0.012 + 0.078 * torch.rand((16, 24), generator=generator)
    v = -0.22 + 0.44 * torch.rand((16, 24), generator=generator)
    features = torch.stack((x, v), dim=-1)
    residual = 800.0 * x.pow(3)
    gain = orthogonal_operator_gain(
        features, residual, operator_library(features), ridge=1.0e-6
    )
    assert float((gain.argmax(dim=-1) == CUBIC_OPERATOR).float().mean()) > 0.9


def test_oracle_proposal_inserts_missing_true_operator():
    proposed = torch.tensor(((0, 2, 3), (0, 1, 2), (1, 3, 5), (0, 3, 4)))
    classes = torch.tensor((1, 2, 0, 3))
    output = _ensure_oracle_proposal(proposed, classes)
    assert CUBIC_OPERATOR in output[0].tolist()
    assert COUPLED_OPERATOR in output[1].tolist()
    torch.testing.assert_close(output[2], proposed[2])
    torch.testing.assert_close(output[3], proposed[3])


def test_operator_truth_indices_are_distinct_and_in_library():
    config = V4Config(episodes=8)
    assert config.proposal_topk < len(OPERATOR_NAMES)
    assert CUBIC_OPERATOR != COUPLED_OPERATOR
    assert 0 <= CUBIC_OPERATOR < len(OPERATOR_NAMES)
    assert 0 <= COUPLED_OPERATOR < len(OPERATOR_NAMES)
