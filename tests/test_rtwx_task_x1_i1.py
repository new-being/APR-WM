"""I1 parameterization identities and CLI."""

from __future__ import annotations

import torch

from aprwm_v0.cli import build_parser
from aprwm_v0.policy.diffusion_parameterization import alpha_sigma, decode_x0_eps, diffusion_target
from aprwm_v0.policy.diffusion_schedule import CosineSchedule


def test_cli_task_x1_i1():
    assert build_parser().parse_args(["rtwx-task-x1-i1"]).command == "rtwx-task-x1-i1"


def _pair():
    sched = CosineSchedule(100, "cpu")
    x0 = torch.randn(4, 8, 14)
    eps = torch.randn(4, 8, 14)
    t = torch.full((4,), 99, dtype=torch.long)
    xt = sched.q_sample(x0, t, eps)
    return sched, x0, eps, t, xt


def test_epsilon_decode_identity():
    sched, x0, eps, t, xt = _pair()
    target = diffusion_target(x0, eps, t, sched, "epsilon")
    assert torch.equal(target, eps)
    x0h, epsh = decode_x0_eps(eps, xt, t, sched, "epsilon")
    assert torch.equal(epsh, eps)
    assert float((x0h - x0).abs().max()) < 2e-3


def test_v_prediction_no_div_alpha():
    sched, x0, eps, t, xt = _pair()
    a, s = alpha_sigma(sched, t, xt)
    v = diffusion_target(x0, eps, t, sched, "v_prediction")
    assert torch.allclose(v, a * eps - s * x0, atol=1e-6)
    x0h, epsh = decode_x0_eps(v, xt, t, sched, "v_prediction")
    assert torch.allclose(x0h, a * xt - s * v, atol=1e-6)
    assert float((x0h - x0).abs().max()) < 1e-4
    assert float((epsh - eps).abs().max()) < 1e-4
    assert float(a.abs().min()) < 1e-3


def test_sample_decode_identity():
    sched, x0, eps, t, xt = _pair()
    target = diffusion_target(x0, eps, t, sched, "sample")
    assert torch.equal(target, x0)
    x0h, epsh = decode_x0_eps(x0, xt, t, sched, "sample")
    assert torch.equal(x0h, x0)
    assert float((epsh - eps).abs().max()) < 2e-3
