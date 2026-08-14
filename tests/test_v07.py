import torch

from aprwm_v0.v06 import _top_budget
from aprwm_v0.v07 import cost_aware_scores, residual_utility, task_weights


def test_utility_is_loss_improvement():
    target = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
    prediction = torch.tensor([[1.5, -0.5], [1.0, 3.0]])
    weights = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    utility = residual_utility(target, prediction, weights)
    before = (weights * target.square()).sum(dim=-1)
    after = (weights * (target - prediction).square()).sum(dim=-1)
    torch.testing.assert_close(utility, before - after)


def test_cost_aware_utility_preserves_ranking_for_constant_cost():
    utility = torch.tensor([-0.4, 0.2, 1.7, 0.1, 0.8])
    direct = _top_budget(utility, 0.4)
    ratio = _top_budget(cost_aware_scores(utility, 8.0, mode="ratio"), 0.4)
    difference = _top_budget(
        cost_aware_scores(utility, 8.0, mode="difference", lambda_cost=0.3), 0.4
    )
    assert torch.equal(direct, ratio)
    assert torch.equal(direct, difference)


def test_task_weights_are_positive_and_anisotropic():
    features = torch.zeros(4, 12)
    features[:, 4] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    features[:, 5] = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    weights = task_weights(features)
    assert bool((weights > 0).all())
    assert not torch.equal(weights[:, 0], weights[:, 1])
