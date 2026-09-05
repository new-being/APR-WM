"""Unit tests for V7C contact-channel slots and H4."""

from __future__ import annotations

import numpy as np

from aprwm_v0.r3_v7c import (
    CONTACT_INSTANT_COL,
    CONTACT_WINDOW_COL,
    HELD_SEEDS,
    h4_sensor_useful,
    sensor_contact_slots,
    zero_contact_slots,
)


def test_none_channel_zeros_only_contact_slots():
    steps = np.arange(24, dtype=np.float64).reshape(2, 12)
    none = zero_contact_slots(steps)
    assert none[0, CONTACT_INSTANT_COL] == 0.0
    assert none[0, CONTACT_WINDOW_COL] == 0.0
    assert none[0, 0] == steps[0, 0]
    assert none[1, 4] == steps[1, 4]


def test_sensor_slots_use_proxy_not_oracle_count():
    steps = np.ones((4, 12))
    steps[:, CONTACT_INSTANT_COL] = 1.0
    proxy = np.asarray([0.0, 0.2, 0.4, 0.8])
    resid = np.asarray([1.0, 1.0, 3.0, 5.0])
    sensor = sensor_contact_slots(steps, proxy, resid, window=2)
    assert abs(sensor[3, CONTACT_INSTANT_COL] - 0.8) < 1.0e-12
    assert abs(sensor[3, CONTACT_WINDOW_COL] - 4.0) < 1.0e-12
    assert HELD_SEEDS == (21131, 21141)


def test_h4_accepts_occupancy_and_c1_or_brier():
    assert h4_sensor_useful(
        brier_mid_s=0.20,
        brier_mid_n=0.10,
        brier_final_s=0.20,
        brier_final_n=0.10,
        b_c0_s=0.10,
        b_c0_n=0.20,
        brier_c1_s=0.15,
        brier_c1_n=0.25,
    )
    assert h4_sensor_useful(
        brier_mid_s=0.10,
        brier_mid_n=0.20,
        brier_final_s=0.11,
        brier_final_n=0.21,
        b_c0_s=0.30,
        b_c0_n=0.10,
        brier_c1_s=0.40,
        brier_c1_n=0.10,
    )
    assert not h4_sensor_useful(
        brier_mid_s=0.20,
        brier_mid_n=0.10,
        brier_final_s=0.20,
        brier_final_n=0.10,
        b_c0_s=0.30,
        b_c0_n=0.10,
        brier_c1_s=0.40,
        brier_c1_n=0.10,
    )
