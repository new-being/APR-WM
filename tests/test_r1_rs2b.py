"""Unit tests for RS2B consequence transport (no detectability retune)."""

from __future__ import annotations

from aprwm_v0.r1_rs2_formal import FORMAL_SEEDS
from aprwm_v0.r1_rs2b import REGIMES, R1RS2BConfig, SCRIPT, _aggregate


def test_rs2b_matrix_includes_c2_and_not_an_excitation_sweep():
    cfg = R1RS2BConfig()
    assert cfg.seeds == FORMAL_SEEDS
    assert REGIMES == ("C0", "C1-L", "C1-H", "C2")
    assert cfg.script == SCRIPT == "fast_pull"
    assert len(cfg.seeds) * len(cfg.regimes) == 20


def test_aggregate_c2_harm_does_not_retune_c_tol():
    c_tol = 0.002
    rows = []
    for seed in FORMAL_SEEDS:
        rows.append(
            {
                "seed": seed,
                "regime": "C0",
                "C_mode_a": 0.0,
                "C_contact": 0.0,
                "L_contact": 0.0004,
                "consequential_mode_a": False,
                "consequential_contact": False,
            }
        )
        rows.append(
            {
                "seed": seed,
                "regime": "C1-H",
                "C_mode_a": 0.004,
                "C_contact": 0.005,
                "L_contact": 0.005,
                "consequential_mode_a": True,
                "consequential_contact": True,
            }
        )
        rows.append(
            {
                "seed": seed,
                "regime": "C2",
                "C_mode_a": 0.0003,
                "C_contact": 0.0001,
                "L_contact": 0.05,
                "consequential_mode_a": False,
                "consequential_contact": False,
            }
        )
    summary = _aggregate(rows, c_tol)
    assert summary["h0_c0_sanity"]["pass"] is True
    assert summary["h2_c2_latch_harm"]["pass"] is True
    assert summary["rs2b_go"] is True
    assert summary["rewrites_rs2_go"] is False
    assert summary["unlocks_c_tol_retune"] is False
    assert summary["reopens_rs2a"] is False
    rows[0]["consequential_mode_a"] = True
    failed = _aggregate(rows, c_tol)
    assert failed["h0_c0_sanity"]["pass"] is False
    assert failed["rs2b_go"] is False
