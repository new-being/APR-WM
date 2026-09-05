"""RTWX-O0BEL0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.geometry.relative_belief import (
    RelativeBeliefState,
    mean_surprise,
    recent_hazard,
    registration_surprise,
    update_belief,
)
from aprwm_v0.geometry.relative_rigid_registration import RelativeICPConfig, estimate_relative_rigid
from aprwm_v0.geometry.relative_state_chain import ChainState, propagate_state
from aprwm_v0.geometry.relative_validity import symmetric_support_score
from aprwm_v0.rtwx_o0bel0 import (
    BELIEF_EPS,
    CAL_SEED,
    FAILURE_HORIZON,
    FORMAL_SEED,
    RECENT_WINDOW,
    RTWXO0BEL0Config,
    _auroc,
    _eval_sequence_bel0,
    _observability,
    _step_hazard_dataset,
    future_failure_labels,
    run_rtwx_o0bel0,
)


def test_cli():
    assert build_parser().parse_args(["rtwx-o0bel0"]).command == "rtwx-o0bel0"


def test_surprise_q1_zero():
    assert registration_surprise(1.0, BELIEF_EPS) == pytest.approx(0.0, abs=1e-9)


def test_surprise_monotone_in_q():
    assert registration_surprise(0.5) > registration_surprise(0.8)


def test_cumulative_surprise_monotone():
    st = RelativeBeliefState()
    prev = 0.0
    for q in [0.9, 0.8, 0.7, 0.6]:
        st = update_belief(st, q, eps=BELIEF_EPS)
        assert st.cumulative_surprise >= prev
        prev = st.cumulative_surprise


def test_mean_surprise_formula():
    assert mean_surprise(10.0, 4) == pytest.approx(2.5)


def test_recent_hazard_window_four():
    hist = [1.0, 0.5, 0.25, 0.1, 0.05]
    assert recent_hazard(hist, window=RECENT_WINDOW) == pytest.approx(
        max(registration_surprise(q, BELIEF_EPS) for q in hist[-4:])
    )


def test_future_label_excludes_current_step():
    unsafe = [False, True, False, False]
    labels = future_failure_labels(unsafe, horizon=2)
    assert labels[0] == 1.0
    assert labels[1] == 0.0


def test_future_label_predicts_ahead():
    unsafe = [False, False, False, True, False]
    labels = future_failure_labels(unsafe, horizon=2)
    assert labels[1] == 1.0


def test_future_label_last_steps_invalid():
    unsafe = [False] * 8
    labels = future_failure_labels(unsafe, horizon=FAILURE_HORIZON)
    assert np.all(np.isnan(labels[-FAILURE_HORIZON:]))


def test_belief_does_not_modify_state():
    st = ChainState(p=np.zeros(3), n=np.array([0.0, 1.0, 0.0]), t_accum=np.eye(4))
    p0, n0 = st.p.copy(), st.n.copy()
    belief = RelativeBeliefState()
    for q in [0.2, 0.3, 0.4]:
        belief = update_belief(belief, q)
    assert np.allclose(st.p, p0)
    assert np.allclose(st.n, n0)


def test_time_and_belief_same_mask_for_auc():
    y = np.array([False, True, False, np.nan], dtype=np.float64)
    mask = np.isfinite(y)
    ds = {
        "U": np.array([1.0, 2.0, 3.0, 4.0]),
        "U_bar": np.array([0.5, 1.0, 1.5, 2.0]),
        "R": np.array([0.1, 0.2, 0.3, 0.4]),
        "t": np.array([1.0, 2.0, 3.0, 4.0]),
        "y": y[mask].astype(bool),
    }
    full = {
        "U": ds["U"][mask],
        "U_bar": ds["U_bar"][mask],
        "R": ds["R"][mask],
        "t": ds["t"][mask],
        "y": ds["y"],
    }
    out = _observability(full)
    assert out["n_samples"] == 3


def test_no_positive_labels_nan_auc():
    ds = {
        "U": np.array([1.0, 2.0]),
        "U_bar": np.array([1.0, 1.0]),
        "R": np.array([1.0, 1.0]),
        "t": np.array([1.0, 2.0]),
        "y": np.array([False, False]),
    }
    out = _observability(ds)
    assert np.isnan(out["auc_cum_surprise"])


def test_frozen_seeds():
    assert CAL_SEED == 37610
    assert FORMAL_SEED == 37611


def test_auroc_perfect():
    scores = np.array([3.0, 2.0, 1.0, 0.0])
    labels = np.array([True, True, False, False])
    assert _auroc(scores, labels) == pytest.approx(1.0, abs=1e-6)


def test_smoke(tmp_path: Path):
    out = run_rtwx_o0bel0(
        tmp_path / "bel0",
        config=RTWXO0BEL0Config(
            smoke=True,
            natural_cache="/root/APR-WM/runs/rtwx_o0e0",
            p0_cache="/root/APR-WM/runs/rtwx_o0e0p0",
            r5_run="/root/APR-WM/runs/rtwx_o0e0r5",
        ),
    )
    assert "pattern" in out
    assert out["pattern"] in {
        "belief_sequence_replication_failure",
        "belief_failure_support_insufficient",
        "relative_belief_insufficient",
        "relative_belief_observable",
    }
