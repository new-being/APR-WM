"""CAP-X1 capacity tests (smoke-scale)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aprwm_v0.cap_x1 import CAPX1Config, mlp_param_count, run_cap_x1


def test_cap_x1_refuses_r10(tmp_path: Path):
    with pytest.raises(RuntimeError, match="r10_c0"):
        run_cap_x1(
            tmp_path / "runs" / "r10_c0" / "fake",
            config=CAPX1Config(require_x0_summary=False, pure_widths=(8,), hybrid_widths=(0,), train_seeds=(101,), epochs=1),
        )


def test_mlp_param_count_monotonic():
    assert mlp_param_count(22, 0) == 0
    assert mlp_param_count(22, 8) < mlp_param_count(22, 16)


@pytest.mark.slow
def test_cap_x1_smoke_requires_x0(tmp_path: Path):
    x0 = Path("runs/cap_x0/formal")
    if not (x0 / "summary.json").is_file():
        pytest.skip("CAP-X0 formal artifacts missing")
    result = run_cap_x1(
        tmp_path / "cap_x1",
        config=CAPX1Config(
            x0_data=str(x0),
            pure_widths=(8,),
            hybrid_widths=(0,),
            train_seeds=(101,),
            epochs=2,
            patience=2,
            plan_tasks=2,
            rollout_episodes=4,
            latency_steps=20,
            train_max_samples=4000,
        ),
    )
    assert result["rho"] == 0.0
    assert result["mu"] == 0.0
    assert "R_P_0" in result["metrics"]
