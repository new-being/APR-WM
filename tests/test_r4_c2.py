"""Unit tests for R4-C2 continuous ridge helpers."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r4_c2 import DELTA_MIN, apply_ridge, fit_ridge, mse, zscore, zscore_fit


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = 0.5 * x[:, 0] - 0.25 * x[:, 1]
    weights, mean, std = fit_ridge(x, y, ridge=1.0e-6)
    pred = apply_ridge(x, weights, mean, std)
    assert mse(pred, y) < 1.0e-6
    assert DELTA_MIN == 0.05


def test_zscore_unit_variance():
    values = np.array([0.0009, 0.0011, 0.0013, 0.0012])
    mean, std = zscore_fit(values)
    z = zscore(values, mean, std)
    assert abs(float(z.mean())) < 1.0e-12
    assert abs(float(z.std()) - 1.0) < 1.0e-12
