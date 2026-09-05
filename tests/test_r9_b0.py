"""Unit tests for R9-B0 passive license revocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.r8_p0 import F_MAX_BENIGN, F_MAX_ID, F_MAX_INVALID
from aprwm_v0.r9_b0 import CUSUM_H, CUSUM_K, K_CUT, REVOKE_M, SEQUENCES, STABLE_MIN, run_r9_b0


def test_r9_b0_freezes_passive_cusum():
    assert K_CUT == 2
    assert REVOKE_M == 1
    assert CUSUM_K == 0.5
    assert CUSUM_H == 4.0
    assert STABLE_MIN == 0.90
    assert SEQUENCES == (("stay", F_MAX_ID), ("benign", F_MAX_BENIGN), ("invalid", F_MAX_INVALID))
    src = Path("aprwm_v0/r9_b0.py").read_text(encoding="utf-8")
    assert "reprobe" in src
    assert "bidirectional_belief" in src
    assert "score_uses_A" in src


def test_r9_b0_locked_without_a0(tmp_path: Path):
    with pytest.raises(RuntimeError, match="locked until R9-A0"):
        run_r9_b0(tmp_path, a0_summary=tmp_path / "missing.json")
