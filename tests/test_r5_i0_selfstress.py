"""Unit tests for R5-I0-SELFSTRESS rest-length preload."""

from __future__ import annotations

from aprwm_v0.r5_i0_selfstress import (
    C_RANGE_MIN,
    D_MACRO_MAX,
    HALF_SPAN,
    K_TENDON,
    RHO_MIN,
    grasp_xml_selfstress,
    rest_length,
)


def test_rest_length_sets_hidden_preload_not_ctrl():
    assert abs(rest_length(0.0) - HALF_SPAN) < 1.0e-12
    assert abs(K_TENDON * (HALF_SPAN - rest_length(8.0)) - 8.0) < 1.0e-12
    xml = grasp_xml_selfstress(lam=8.0)
    assert "f_y" in xml
    assert xml.count("<motor") == 1
    assert "springlength" in xml
    assert D_MACRO_MAX == 0.05
    assert RHO_MIN == 0.80
    assert C_RANGE_MIN == 0.10
