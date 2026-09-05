"""PLAN-X1.5 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aprwm_v0.plan_x15 import PLANX15Config, mixture_nll, run_plan_x15, sample_mix


def test_mixture_nll_single_component_matches_gaussian():
    a = torch.zeros(4, 60)
    logits = torch.zeros(4, 4)
    mu = torch.zeros(4, 4, 60)
    ls = torch.zeros(4, 4, 60)
    nll = mixture_nll(a, logits, mu, ls)
    # Equal mixture of identical N(0,1) is still N(0,1).
    expected = 0.5 * (0.0 + 0.0 + np.log(2.0 * np.pi)) * 60
    assert torch.allclose(nll, torch.full_like(nll, expected), atol=1e-4)


def test_sample_mix_shape():
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    mu = np.zeros((4, 60))
    sig = np.ones((4, 60)) * 0.1
    rng = np.random.default_rng(0)
    a = sample_mix(pi, mu, sig, 12, rng)
    assert a.shape == (12, 60)
    assert np.all(np.abs(a) <= 1.5 + 1e-12)


def test_plan_x15_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_plan_x15(tmp_path / "runs" / "r10_c0" / "x", config=PLANX15Config(max_eval_conditions=1, epochs=1))


@pytest.mark.slow
def test_plan_x15_smoke(tmp_path: Path):
    x0 = Path("runs/plan_x0/formal")
    if not (x0 / "summary.json").is_file():
        pytest.skip("PLAN-X0 formal missing")
    r = run_plan_x15(
        tmp_path / "plan_x15",
        config=PLANX15Config(
            x0_data=str(x0),
            epochs=2,
            patience=2,
            train_seeds=(301,),
            max_eval_conditions=4,
            skip_cem=True,
        ),
    )
    assert r["diffusion"] is False
    assert r["uses_hessian"] is False
    assert "mix" in r["metrics"]["AUC_S"]
