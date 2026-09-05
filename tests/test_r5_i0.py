"""Unit tests for R5-I0 mean-preserving slip-margin μ and prestress X_stat."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r4_i0 import MU
from aprwm_v0.r5_i0 import (
    D_MACRO_MAX,
    RHO_MIN,
    SOLREF0,
    grasp_xml_prestress,
    mean_solref_kt,
    mu_columns,
    quadrupole_modes,
    x_stat_xhalves,
)


def test_mu_columns_preserve_mean():
    cols = mu_columns(0.12)
    assert abs(sum(cols) / len(cols) - MU) < 1.0e-12
    assert cols[0] == MU + 0.12
    assert cols[-1] == MU - 0.12
    assert D_MACRO_MAX == 0.05
    assert RHO_MIN == 0.80


def test_prestress_xml_keeps_r4_taxel_names():
    xml = grasp_xml_prestress()
    assert 'name="pad_left_0_0"' in xml
    assert "pad_left_n_" not in xml
    assert xml.count("<motor") == 6


def test_x_stat_xhalves_is_intra_pad_not_finger():
    ty = np.zeros((2, 4, 4))
    ty[:, :, :2] = 1.0
    ty[:, :, 2:] = -1.0
    assert x_stat_xhalves({"tau_y": ty}) == 32.0
    swap = np.zeros((2, 4, 4))
    swap[0] = 1.0
    swap[1] = -1.0
    assert x_stat_xhalves({"tau_y": swap}) == 0.0


def test_quadrupole_modes_are_macro_nullspace():
    phi1, phi2 = quadrupole_modes(4)
    xs = np.linspace(-1.5, 1.5, 4)
    ix = xs[np.newaxis, :]
    iz = xs[:, np.newaxis]
    for phi in (phi1, phi2):
        assert abs(float(np.mean(phi))) < 1.0e-12
        assert abs(float(np.sum(ix * phi))) < 1.0e-10
        assert abs(float(np.sum(iz * phi))) < 1.0e-10
    assert abs(float(np.sum(phi1 * phi2))) < 1.0e-10
    assert abs(mean_solref_kt(0.0) - SOLREF0) < 1.0e-12
    assert abs(mean_solref_kt(0.0) - mean_solref_kt(0.5 * np.pi)) < 1.0e-12
