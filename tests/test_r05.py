import math

import torch

from aprwm_v0.r05 import V6R05Config, pairwise_auroc


def test_pairwise_auroc_corrects_ties():
    scores = torch.tensor([1.0, 1.0, 0.0, 2.0])
    unsafe = torch.tensor([True, False, False, True])
    assert math.isclose(pairwise_auroc(scores, unsafe), 0.875)


def test_r05_reuses_disjoint_r04_cohorts():
    config = V6R05Config()
    assert set(config.selection_seeds).isdisjoint(config.confirmation_seeds)
