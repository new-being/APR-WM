"""TASK-X1-I0 unit tests (no RoboTwin)."""

from __future__ import annotations

import torch

from aprwm_v0.cli import build_parser
from aprwm_v0.evaluation.task_x1_i0 import c0_oracle_reverse, c1_oracle_x0
from aprwm_v0.policy.diffusion_schedule import CosineSchedule, TRAIN_STEPS


def test_cli_i0():
    assert build_parser().parse_args(["rtwx-task-x1-i0"]).command == "rtwx-task-x1-i0"


def test_oracle_reverse_near_identity():
    sched = CosineSchedule(TRAIN_STEPS, "cpu")
    torch.manual_seed(1)
    x0 = torch.randn(2, 8, 14)
    eps = torch.randn_like(x0)
    hat = c0_oracle_reverse(x0, eps, sched)
    assert float((hat - x0).abs().max()) < 1e-3
    hat50 = c1_oracle_x0(x0, eps, 50, sched)
    assert float((hat50 - x0).abs().max()) < 1e-5
