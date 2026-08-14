import torch

from aprwm_v0.r04a import V6R04AConfig, _episode_rmse


def test_r04a_seeds_are_new_held_out_seeds():
    assert set(V6R04AConfig().seeds).isdisjoint({13, 23, 33, 1001, 1011, 1021, 1031, 1041})


def test_r04a_defaults_use_confirmatory_not_selection_cohort():
    assert set(V6R04AConfig().seeds).isdisjoint({2001, 2011, 2021, 2031, 2041})


def test_episode_rmse_masks_invalid_queries():
    prediction = torch.tensor([[[[1.0, 2.0]], [[100.0, 200.0]]]])
    truth = torch.zeros_like(prediction)
    validity = torch.tensor([[[True], [False]]])
    result = _episode_rmse(prediction, truth, validity)
    torch.testing.assert_close(result, torch.ones((1, 1)))
