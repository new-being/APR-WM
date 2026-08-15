"""Unit tests for R1-RS1A detectors and ROC helpers."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r1_rs1a import (
    _auprc,
    _auroc,
    _operating_point,
    _recall_at_threshold,
)


def test_auroc_perfect_and_chance():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert _auroc(scores, labels) == 1.0
    chance = _auroc(np.asarray([0.5, 0.5, 0.5, 0.5]), labels)
    assert abs(chance - 0.5) < 1e-9


def test_auprc_perfect():
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    assert _auprc(scores, labels) == 1.0


def test_operating_point_respects_fpr_cap():
    # Two negatives at low scores, positives higher
    scores = np.asarray([0.0, 0.05, 0.2, 0.4, 0.6, 0.8])
    labels = np.asarray([0, 0, 1, 1, 1, 1])
    op = _operating_point(scores, labels, target_fpr=0.0)
    assert op["fpr"] == 0.0
    assert op["tpr"] >= 0.0
    assert _recall_at_threshold(scores[labels == 1], np.ones(4, dtype=np.int32), op["threshold"]) == op["tpr"]
