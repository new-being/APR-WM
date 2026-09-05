"""AC1-D1 unit tests (no RoboTwin)."""

from __future__ import annotations

import numpy as np

from aprwm_v0.cli import build_parser
from aprwm_v0.evaluation.state_support_d1 import GlobalMahalanobis, PhaseConditionedKNN, phase_from_process
from aprwm_v0.rtwx_ac1_d1 import _select


def test_cli_d1():
    assert build_parser().parse_args(["rtwx-ac1-d1"]).command == "rtwx-ac1-d1"


def test_phase_no_time():
    assert phase_from_process(np.array([0.5, 0.0, 1.0, 0.0, 1.0])) == 3
    assert phase_from_process(np.array([0.05, 1.0, 0.4, 1.0, 0.0])) == 1
    assert phase_from_process(np.array([0.4, 0.0, 0.4, 0.0, 0.0])) == 0
    assert phase_from_process(np.array([0.05, 0.0, 0.2, 0.0, 0.0])) == 2


def test_mahalanobis_inlier_outlier():
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(80, 6))
    mu, sig = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    m = GlobalMahalanobis(tr, mu, sig)
    d_in = m.distances(tr[:10])
    d_out = m.distances(tr[:10] + 8.0)
    assert float(np.median(d_out)) > float(np.median(d_in))


def test_select_requires_all_tasks():
    def rec(r):
        return {"r_ood": r}

    bad = {
        "B0": {"place_empty_cup": rec(0.01), "put_object_cabinet": rec(0.21), "stamp_seal": rec(0.0)},
        "B1": {"place_empty_cup": rec(0.01), "put_object_cabinet": rec(0.05), "stamp_seal": rec(0.02)},
        "B2": {"place_empty_cup": rec(0.01), "put_object_cabinet": rec(0.04), "stamp_seal": rec(0.02)},
        "B3": {"place_empty_cup": rec(0.01), "put_object_cabinet": rec(0.04), "stamp_seal": rec(0.02)},
    }
    sel = _select(bad)
    assert sel["winner"] == "B2"
    none = {
        b: {"place_empty_cup": rec(0.2), "put_object_cabinet": rec(0.2), "stamp_seal": rec(0.2)} for b in ("B0", "B1", "B2", "B3")
    }
    assert _select(none)["winner"] is None
    _ = PhaseConditionedKNN
