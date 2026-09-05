"""Unit tests for R1-RS1A.2 weak-signal statistics."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r1_rs1a2 import _compute_statistics


def test_statistics_on_aligned_operator_signal():
    # Synthetic: r = alpha * |v| v with small noise; physical (x,v)
    rng = np.random.default_rng(0)
    v = rng.uniform(-0.5, 0.5, size=32)
    x = rng.uniform(0.05, 0.3, size=32)
    alpha = -0.12
    r = alpha * np.abs(v) * v + rng.normal(0.0, 1e-4, size=32)
    physical = np.stack([x, v], axis=1)
    stats = _compute_statistics(r, physical, u=0.0)
    assert stats["D0"] > 0
    assert stats["D_lib"] > 0
    assert stats["lib_best_operator"] == "abs_v_v"
    assert stats["D_support"] < 1e-6


def test_direction_and_corr_for_constant_sign():
    r = np.asarray([0.01, 0.02, 0.015, 0.012], dtype=np.float64)
    physical = np.stack([np.full(4, 0.1), np.full(4, 0.2)], axis=1)
    stats = _compute_statistics(r, physical, u=0.0)
    assert abs(stats["D_dir"] - 1.0) < 1e-5
    assert stats["D_corr"] > 0.99
