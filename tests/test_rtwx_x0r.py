"""RoboTwin-X0R tests (no full SAPIEN / official task required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aprwm_v0.cli import build_parser
from aprwm_v0.rtwx_x0r import (
    DELTA_ID,
    EPS_EXC,
    PHI_HARD,
    TAU1,
    TAU10,
    RTWX0RConfig,
    excitation_score,
    fit_phi,
    run_rtwx_x0r,
    score_gates,
)


def test_cli_has_rtwx_x0r():
    parser = build_parser()
    args = parser.parse_args(["rtwx-x0r", "--output", "runs/rtwx_x0r"])
    assert args.command == "rtwx-x0r"


def test_x0r_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_rtwx_x0r(tmp_path / "runs" / "r10_c0" / "x", config=RTWX0RConfig(backend="numpy", n_train_ep=2, n_test_ep=2, n_steps=12, latent_epochs=1))


def test_gates_frozen_outside_x0_copy_regime():
    assert EPS_EXC >= 0.05
    assert DELTA_ID >= 0.02
    assert TAU1 != pytest.approx(0.00326, abs=1e-4)
    assert TAU1 != pytest.approx(0.158, abs=1e-3)
    assert TAU10 < 20.0


def test_header_written_with_frozen_eps(tmp_path: Path):
    out = tmp_path / "x0r"
    cfg = RTWX0RConfig(backend="numpy", n_train_ep=6, n_test_ep=4, n_steps=40, n_cal_steps=12, latent_epochs=8, latent_widths=(8, 16), seed=3)
    summary = run_rtwx_x0r(out, config=cfg)
    header = (out / "run.header.txt").read_text(encoding="utf-8")
    assert "eps_exc=0.08" in header
    assert "delta_id=0.05" in header
    assert "tau1=0.8" in header or "tau1=0.80" in header
    assert "capacity_claim=false" in header
    assert summary["capacity_claim"] is False
    assert summary["unlocks_r10_c0"] is False
    assert (out / "trajectories.npz").is_file()
    assert summary["pattern"] in {
        "instrument_ready",
        "excitation_failure",
        "not_force_auditable",
        "phi_unidentified",
        "latent_rollout_unstable",
    }
    assert summary["rtwx_x0r_passed"] == (summary["pattern"] == "instrument_ready")
    assert summary["rtwx_x0r_passed"] == summary["instrument_ready"]


def test_g1_fail_is_not_force_auditable():
    rng = np.random.default_rng(0)
    n_ep, n_steps, d = 3, 30, 2
    s = rng.normal(size=(n_ep * n_steps, d))
    pool = {"s": s, "sp": s + 0.2, "u": rng.normal(size=(n_ep * n_steps, 1)), "phi": np.tile(PHI_HARD, (n_ep * n_steps, 1)), "n_steps": n_steps, "n_ep": n_ep}
    cfg = RTWX0RConfig(n_cal_steps=8)
    scored = score_gates(
        cfg=cfg,
        train=pool,
        test=pool,
        latent_curve=[{"hidden": 16, "E1": 0.1, "E_roll10": 0.2}, {"hidden": 32, "E1": 0.05, "E_roll10": 0.3}],
        g1_audit={"has_q": True, "has_qdot": True, "has_tau": False, "has_qddot_fd": True, "n_dof": 1},
    )
    assert scored["G1"] is False
    assert scored["pattern"] == "not_force_auditable"


def test_g3_forbids_best_e1_equals_worst_rollout():
    rng = np.random.default_rng(1)
    n_ep, n_steps = 4, 40
    s = rng.normal(size=(n_ep * n_steps, 2)) * 0.5
    sp = s + 0.4
    pool = {
        "s": s,
        "sp": sp,
        "u": rng.normal(size=(n_ep * n_steps, 1)),
        "phi": np.tile([1.5, 0.08, 0.5], (n_ep * n_steps, 1)),
        "n_steps": n_steps,
        "n_ep": n_ep,
        "audit": {"has_q": True, "has_qdot": True, "has_tau": True, "has_qddot_fd": True, "n_dof": 1},
    }
    cfg = RTWX0RConfig(n_cal_steps=10)
    curve = [
        {"hidden": 16, "E1": 0.4, "E_roll10": 0.5},
        {"hidden": 128, "E1": 0.05, "E_roll10": 9.0},
    ]
    scored = score_gates(cfg=cfg, train=pool, test=pool, latent_curve=curve, g1_audit=pool["audit"])
    assert scored["metrics"]["g3"]["best_e1_is_worst_rollout"] is True
    assert scored["G3"] is False


def test_fit_phi_recovers_mass_on_numpy_plant():
    from aprwm_v0.rtwx_x0r import collect_numpy, dt_eff

    cfg = RTWX0RConfig(backend="numpy", n_train_ep=8, n_test_ep=4, n_steps=60, seed=11)
    pool = collect_numpy(cfg, train=True, rng=np.random.default_rng(11))
    hat = fit_phi(pool["s"], pool["u"], pool["sp"], dt_eff(cfg))
    assert hat[0] == pytest.approx(np.mean(pool["phi"][:, 0]), rel=0.35)


def test_excitation_detects_motion():
    n_ep, n_steps = 2, 30
    t = np.linspace(0, 1, n_steps)
    q = np.tile(t, n_ep)
    s = np.stack([q, np.ones_like(q)], axis=1)
    assert excitation_score(s, n_steps, 10) > 0.08
