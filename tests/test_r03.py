import torch

from aprwm_v0.r03 import V6R03Config, _masked_rmse


def test_formal_seeds_are_disjoint_from_development_seeds():
    assert set(V6R03Config().seeds).isdisjoint({13, 23, 33})


def test_validity_mask_excludes_out_of_support_error():
    error = torch.tensor(((1.0, 1.0), (100.0, 100.0)))
    mask = torch.tensor(((True, True), (False, False)))
    assert _masked_rmse(error, mask) == 1.0
