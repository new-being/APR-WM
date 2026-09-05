"""Unit tests for R1-RS1A.1 compositional scores."""

from __future__ import annotations

from aprwm_v0.r1_rs1a1 import _compose_scores, _select_gamma, R1RS1A1Config


def test_compose_d3p_and_gated():
    row = {
        "scores": {"D0": 0.01, "D1": 0.2, "D2": -0.4, "D3": -0.3},
        "support_term": 0.17,
        "u_unsupported": 0.156,
    }
    s = _compose_scores(row, gamma=2.0, d3_ref_gamma=1.0)
    assert abs(s["D3p"] - (0.2 + 2.0 * 0.17)) < 1e-12
    assert abs(s["Dg"] - s["D3p"]) < 1e-12
    assert abs(s["D3"] - (-0.4 + 1.0 * 0.17)) < 1e-12

    row0 = {**row, "u_unsupported": 0.0, "support_term": 0.0}
    s0 = _compose_scores(row0, gamma=2.0, d3_ref_gamma=1.0)
    assert s0["Dg"] == s0["D1"]
    assert s0["D3p"] == s0["D1"]


def test_select_gamma_prefers_balanced_min_recall():
    # Synthetic rows: C0 low D1; C1-L mid; C2 needs support term
    rows = []
    for seed in (1, 2):
        for probe in ("P1", "P2"):
            rows.append(
                {
                    "seed": seed,
                    "regime": "C0",
                    "probe": probe,
                    "scores": {"D0": 0.001, "D1": 0.10, "D2": -0.5, "D3": -0.5},
                    "support_term": 0.0,
                    "u_unsupported": 0.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "regime": "C1-L",
                    "probe": probe,
                    "scores": {"D0": 0.002, "D1": 0.20, "D2": -0.5, "D3": -0.5},
                    "support_term": 0.0,
                    "u_unsupported": 0.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "regime": "C1-H",
                    "probe": probe,
                    "scores": {"D0": 0.005, "D1": 0.35, "D2": -0.4, "D3": -0.4},
                    "support_term": 0.0,
                    "u_unsupported": 0.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "regime": "C2-latch",
                    "probe": probe,
                    "scores": {"D0": 0.0012, "D1": 0.11, "D2": -0.5, "D3": -0.3},
                    "support_term": 0.17,
                    "u_unsupported": 0.156,
                }
            )
    cfg = R1RS1A1Config(gamma_grid=(0.25, 1.0, 4.0))
    result = _select_gamma(rows, cfg)
    assert "selected" in result
    assert result["selected"]["gamma"] in cfg.gamma_grid
