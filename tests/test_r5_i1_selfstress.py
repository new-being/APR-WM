"""Unit tests for R5-I1-SELFSTRESS locks."""

from __future__ import annotations

from aprwm_v0.r5_i0_selfstress import Y_KEYS
from aprwm_v0.r5_i1_selfstress import DELTA_REL_MIN, HELD_LAM, TRAIN_LAM, Y_IDX


def test_i1_split_interpolates_and_y_omits_command_and_x():
    assert TRAIN_LAM == (0.0, 4.0, 8.0, 12.0)
    assert HELD_LAM == (2.0, 6.0, 10.0)
    assert DELTA_REL_MIN == 0.05
    names = [Y_KEYS[i] for i in Y_IDX]
    assert names == ["q", "v", "a", "tau_motor"]
    assert "u" not in names
    assert "x_stat" not in names
    assert "t_left" not in names
