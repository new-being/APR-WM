"""RTWX-O0D tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_o0d import (
    E_P_MEM_MAX,
    MEM_N,
    SEED_AUDIT,
    architecture_audit,
    RTWXO0DConfig,
    run_rtwx_o0d,
    _blob_frame,
    _fit_z2d,
    _pred_z2d,
    _z2d_from_mask,
)


def test_cli_has_rtwx_o0d():
    args = build_parser().parse_args(["rtwx-o0d", "--output", "runs/rtwx_o0d"])
    assert args.command == "rtwx-o0d"


def test_o0d_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_o0d(tmp_path / "runs" / "r10_c0" / "x", config=RTWXO0DConfig(backend="numpy", smoke=True))


def test_o0d_frozen():
    assert SEED_AUDIT == 17601
    assert MEM_N == 256
    assert E_P_MEM_MAX == 0.15
    arch = architecture_audit()
    assert arch["uses_global_average_pool"] is False
    assert arch["spatial_map_then_flatten"] is True


def test_z2d_lstsq_recovers_blob_position():
    rng = np.random.default_rng(0)
    ps, zs = [], []
    for _ in range(40):
        p = rng.uniform([-0.3, -0.2, 0.7], [0.3, 0.05, 0.72])
        _, mask = _blob_frame(p, 32)
        z, vis = _z2d_from_mask(mask)
        assert vis > 0
        ps.append(p)
        zs.append(z)
    p = np.stack(ps)
    z = np.stack(zs)
    w = _fit_z2d(z, p)
    hat = _pred_z2d(w, z)
    err = np.linalg.norm(hat - p, axis=1).mean()
    assert err < 0.05


def test_o0d_numpy_smoke(tmp_path: Path):
    out = tmp_path / "o0d"
    result = run_rtwx_o0d(
        out,
        config=RTWXO0DConfig(
            backend="numpy",
            smoke=True,
            n_train_ep=3,
            n_val_ep=2,
            n_test_ep=2,
            n_steps=12,
            seed=13,
            vis_epochs=4,
            mem_n=24,
            mem_epochs=20,
            rgb_size=32,
        ),
    )
    assert result["does_not_unlock_o1"] is True
    assert result["pattern"] in {
        "perception_instrument_failure",
        "localization_bottleneck",
        "pose_geometry_unresolved",
    }
    assert result["D3"]["D3_ok"] is True
    assert (out / "overlays").is_dir()
