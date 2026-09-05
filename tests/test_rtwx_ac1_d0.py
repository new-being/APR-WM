"""AC1-D0 unit tests (no RoboTwin)."""

from __future__ import annotations

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.evaluation.state_coverage_metrics import auroc_higher_is_policy, first_persistent_exit, ood_occupancy, tau_95
from aprwm_v0.evaluation.state_support import TaskConditionedStateSupport


def test_cli_d0():
    assert build_parser().parse_args(["rtwx-ac1-d0"]).command == "rtwx-ac1-d0"


def test_knn_and_tau():
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(40, 8))
    mu, sig = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    sup = TaskConditionedStateSupport(tr, mu, sig, k=5)
    d_tr = sup.distances(tr)
    assert d_tr.shape == (40,)
    assert float(np.median(d_tr)) < float(sup.distance(np.ones(8) * 8.0))
    tau = tau_95(d_tr)
    assert ood_occupancy(d_tr, tau) <= 0.12


def test_persist_and_auroc():
    d = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    assert first_persistent_exit(d, 0.5, persist=3) == 2
    assert first_persistent_exit(d, 0.5, persist=4) is None
    pol = np.array([2.0, 3.0, 4.0])
    exp = np.array([0.1, 0.2, 0.15])
    assert auroc_higher_is_policy(pol, exp) >= 0.99
    assert auroc_higher_is_policy(exp, pol) <= 0.01
