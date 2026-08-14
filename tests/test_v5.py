import torch

from aprwm_v0.v5 import (
    V5Config,
    _update_candidate_belief,
    lower_confidence_acceptance,
    noise_normalized_action_score,
)


def test_normalized_score_prefers_separated_low_uncertainty_action():
    means = torch.tensor([[[0.0, 2.0], [0.0, 1.0]]])
    variances = torch.tensor([[[0.1, 0.1], [0.1, 0.1]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    score = noise_normalized_action_score(
        means, variances, probabilities, 0.1, weighted=False
    )
    assert score[0, 0] > score[0, 1]


def test_normalized_score_penalizes_predictive_uncertainty():
    means = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    variances = torch.tensor([[[0.01, 0.01], [2.0, 2.0]]])
    probabilities = torch.tensor([[0.5, 0.5]])
    score = noise_normalized_action_score(
        means, variances, probabilities, 0.1, weighted=False
    )
    assert score[0, 0] > score[0, 1]


def test_candidate_update_normalizes_and_favors_matching_model():
    means = torch.tensor([[[0.0], [1.0]]])
    covariance = torch.eye(1).view(1, 1, 1, 1).repeat(1, 2, 1, 1) * 0.01
    log_probabilities = torch.log(torch.tensor([[0.5, 0.5]]))
    design = torch.ones((1, 2, 1))
    _, _, updated_log = _update_candidate_belief(
        means,
        covariance,
        log_probabilities,
        design,
        torch.tensor([1.0]),
        0.01,
        torch.tensor([True]),
    )
    probabilities = updated_log.exp()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))
    assert probabilities[0, 1] > probabilities[0, 0]


def test_lcb_acceptance_requires_stable_positive_improvement():
    config = V5Config(
        acceptance_rmse=1.0,
        acceptance_complexity_price=0.0,
        lcb_z=1.0,
    )
    incumbent = torch.ones((2, 8))
    candidate = torch.stack((torch.full((8,), 0.2), torch.full((8,), 1.2)))
    accepted, lcb = lower_confidence_acceptance(
        incumbent, candidate, torch.zeros(2), config
    )
    assert bool(accepted[0])
    assert not bool(accepted[1])
    assert lcb[0] > lcb[1]
