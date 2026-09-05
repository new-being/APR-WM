"""RTWX-X0ES2 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0es import _dphi_dx
from aprwm_v0.rtwx_x0es2 import RTWX0ES2Config, _dphi_torch, run_rtwx_x0es2


def test_cli_has_rtwx_x0es2():
    args = build_parser().parse_args(["rtwx-x0es2", "--output", "runs/rtwx_x0es2"])
    assert args.command == "rtwx-x0es2"


def test_x0es2_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0es2(
            tmp_path / "runs" / "r10_c0" / "x",
            config=RTWX0ES2Config(backend="numpy", smoke=True, old_cache=str(tmp_path / "no.npz")),
        )


def test_jacobian_torch_matches_numpy():
    rng = np.random.default_rng(0)
    d = 12
    q = rng.normal(size=d)
    qd = rng.normal(size=d)
    eq = rng.normal(size=d)
    jn = _dphi_dx(q, qd, eq)
    qt = torch.tensor(q[None], dtype=torch.float64)
    jt = _dphi_torch(qt, torch.tensor(qd[None], dtype=torch.float64), torch.tensor(eq[None], dtype=torch.float64))
    np.testing.assert_allclose(jt[0].numpy(), jn, rtol=1e-12, atol=1e-12)
