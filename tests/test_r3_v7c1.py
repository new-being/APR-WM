"""Unit tests for V7C.1 tactile encoder fusion and taxel binning."""

from __future__ import annotations

import numpy as np
import torch

from aprwm_v0.r1_rs2 import bin_taxel_force
from aprwm_v0.r3_v7c import CONTACT_INSTANT_COL, CONTACT_WINDOW_COL, zero_contact_slots
from aprwm_v0.r3_v7c1 import HELD_SEEDS, TactileEpi, Z_DIM


def test_taxel_bin_is_local_grid():
    assert bin_taxel_force(np.asarray([0.0, 0.0]), 1.0) is not None
    assert bin_taxel_force(np.asarray([1.0, 0.0]), 1.0) is None
    assert HELD_SEEDS == (22131, 22141)


def test_encoder_writes_only_contact_slots():
    model = TactileEpi(hidden=8)
    steps = torch.randn(1, 4, 12)
    taxel = torch.rand(1, 4, 8, 8)
    fused = model.fuse(steps, taxel)
    assert fused.shape == steps.shape
    assert fused[0, 0, :CONTACT_INSTANT_COL].equal(steps[0, 0, :CONTACT_INSTANT_COL])
    assert fused[0, 0, CONTACT_WINDOW_COL + 1 :].equal(steps[0, 0, CONTACT_WINDOW_COL + 1 :])
    z = model.encode(taxel)
    assert z.shape == (1, 4, Z_DIM)
    none = zero_contact_slots(steps[0].numpy())
    assert none[0, CONTACT_INSTANT_COL] == 0.0
