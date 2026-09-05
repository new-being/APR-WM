"""SYM-X1 smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.symx_plant import d_state, make_bodies, rollout
from aprwm_v0.symx_x0 import sample_action, sample_state
from aprwm_v0.symx_x1 import (
    SEEDS,
    SYMX1Config,
    _g_star,
    canon_state,
    d_G_np,
    n_axis,
    pack_state,
    run_symx_x1,
    yaw_cs,
)


def test_cli():
    assert build_parser().parse_args(["sym-x1"]).command == "sym-x1"


def test_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_symx_x1(tmp_path / "runs" / "r10_c0" / "x", config=SYMX1Config(smoke=True, x0_summary="runs/symx_x0"))


def test_frozen_seeds():
    assert SEEDS == (41101, 41102, 41103)


def test_dG_le_d():
    body = make_bodies()["C0"]
    rng = np.random.default_rng(0)
    s0 = sample_state(rng, airborne=False)
    a = sample_action(rng)
    tr = rollout(body, s0, a, 4, 0.005)
    G = _g_star("C0")
    d = d_state(tr[0], tr[-1])
    dg = float(d_G_np(tr[0]["p"], tr[0]["R"], tr[0]["v"], tr[0]["w"], tr[-1]["p"], tr[-1]["R"], tr[-1]["v"], tr[-1]["w"], G))
    assert dg <= d + 1e-9


def test_c0_section_kills_yaw():
    G = _g_star("C0")
    rng = np.random.default_rng(1)
    feat, yaw, ax = [], [], []
    for _ in range(80):
        s = sample_state(rng, airborne=False)
        sc, _ = canon_state(s, G)
        feat.append(pack_state(sc))
        yaw.append(yaw_cs(s["R"]))
        ax.append(n_axis(s["R"]))
    feat, yaw, ax = np.stack(feat), np.stack(yaw), np.stack(ax)
    from aprwm_v0.symx_x1 import ridge_r2, axis_r2

    assert ridge_r2(feat[:50], yaw[:50], feat[50:], yaw[50:]) < 0.15
    assert axis_r2(feat[:50], ax[:50], feat[50:], ax[50:]) >= 0.80


def test_numpy_smoke(tmp_path: Path):
    x0 = Path("runs/symx_x0")
    if not (x0 / "summary.json").is_file():
        pytest.skip("X0 summary missing")
    r = run_symx_x1(tmp_path / "sx1", config=SYMX1Config(smoke=True, x0_summary=str(x0)))
    assert r["unlocks_o1"] is False
    assert r["unlocks_symx3"] is False
    assert r["header"]["no_rgb"] is True
    assert r["pattern"] in {
        "instrument_failure",
        "quotient_hurts_prediction",
        "discovered_ne_oracle",
        "false_compress_C4",
        "gauge_still_in_quotient",
        "quotient_utility_supported",
    }
    assert "C0" in r["per_condition"] and "C4" in r["per_condition"]
