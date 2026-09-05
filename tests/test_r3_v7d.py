"""Unit tests for V7D seeds, encoder, and GO conjunction."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from aprwm_v0.r3_v7c5 import temporal_gate
from aprwm_v0.r3_v7d import (
    HELD_SEEDS,
    SEEDS,
    TRAIN_SEEDS,
    VAL_SEED,
    Z_DIM,
    FrozenRGBEncoder,
    _require_v7c5,
)


def test_fresh_seeds():
    assert SEEDS == (27101, 27111, 27121, 27131, 27141)
    assert TRAIN_SEEDS == (27101, 27111)
    assert VAL_SEED == 27121
    assert HELD_SEEDS == (27131, 27141)


def test_encoder_shape_and_capacity():
    enc = FrozenRGBEncoder()
    rgb = torch.zeros(2, 5, 3, 32, 32)
    z = enc(rgb)
    assert tuple(z.shape) == (2, 5, Z_DIM)
    convs = [m for m in enc.net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert convs[0].in_channels == 3 and convs[0].out_channels == 8
    assert convs[1].in_channels == 8 and convs[1].out_channels == 8
    linear = [m for m in enc.net.modules() if isinstance(m, torch.nn.Linear)][0]
    assert linear.in_features == 32 and linear.out_features == 2


def test_go_is_conjunction_and_temporal_gate():
    h1 = h2 = h3 = h4 = True
    h5 = temporal_gate(0.02, 0.01, 0.005)
    assert bool(h1 and h2 and h3 and h4 and h5)
    assert not temporal_gate(-0.046, -0.041, 0.005)
    assert not (h1 and h2 and h3 and h4 and False)


def test_unlock_is_scientific_result_not_c5_go(tmp_path: Path):
    payload = {
        "stage": "R3-V7C.5",
        "smoke": False,
        "scientific_result": True,
        "v7c5_go": False,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = _require_v7c5(path)
    assert loaded["v7c5_go"] is False
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps({"stage": "R3-V7C.5", "smoke": True, "scientific_result": False}),
        encoding="utf-8",
    )
    try:
        _require_v7c5(smoke)
        raise AssertionError("smoke summary must not unlock V7D")
    except RuntimeError:
        pass
