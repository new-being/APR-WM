"""TASK-X1 unit tests: shape, determinism, eta=0, no process, fixed K."""

from __future__ import annotations

import numpy as np
import torch

from aprwm_v0.cli import build_parser
from aprwm_v0.policy.deterministic_ddim import ddim_sample, ddim_time_grid
from aprwm_v0.policy.diffusion_chunk_policy import DiffusionChunkPolicy, policy_seed
from aprwm_v0.policy.diffusion_denoiser_mlp import DiffusionChunkDenoiser
from aprwm_v0.policy.diffusion_schedule import INFER_STEPS, TRAIN_STEPS, CosineSchedule
from aprwm_v0.policy.task_embedding import STATE_DIM


def test_cli_task_x1():
    assert build_parser().parse_args(["rtwx-task-x1"]).command == "rtwx-task-x1"
    ns = build_parser().parse_args(["rtwx-task-x1", "--stage-b-only", "--b1-ckpt", "runs/rtwx_task_x1_i1/B2/best.pt"])
    assert ns.stage_b_only is True
    assert ns.b1_ckpt.endswith("B2/best.pt")


def test_stage_b_gates_frozen():
    from aprwm_v0.rtwx_task_x1 import stage_b_verdict

    s0 = {"pooled": 0.0, "safety": 0.0, "n_tasks_gt0": 0}
    s1_fail = {"pooled": 0.10, "safety": 0.0, "n_tasks_gt0": 1}
    p, w, *_ = stage_b_verdict(s0, s1_fail)
    assert p == "diffusion_chunk_breadth_insufficient"
    assert w is None
    s1_ok = {"pooled": 0.25, "safety": 0.0, "n_tasks_gt0": 2}
    p, w, d, ds = stage_b_verdict(s0, s1_ok)
    assert p == "diffusion_chunk_breadth_supported"
    assert w == "B1"
    assert abs(d - 0.25) < 1e-9
    assert abs(ds) < 1e-9


def test_schedule_and_ddim_grid():
    assert TRAIN_STEPS == 100
    assert INFER_STEPS == 10
    g = ddim_time_grid(100, 10)
    assert len(g) == 10
    assert g[0] == 99
    assert g[-1] == 0


def test_denoiser_shape_no_process():
    m = DiffusionChunkDenoiser(STATE_DIM)
    s = torch.zeros(2, STATE_DIM)
    a = torch.zeros(2, 8, 14)
    t = torch.tensor([3, 8])
    y = m(a, t, s, torch.tensor([0, 1]))
    assert y.shape == (2, 8, 14)
    assert not hasattr(m, "process")


def test_ddim_deterministic():
    m = DiffusionChunkDenoiser(STATE_DIM)
    m.eval()
    sched = CosineSchedule(TRAIN_STEPS, "cpu")
    pol = DiffusionChunkPolicy(m, sched, "cpu")
    st = np.zeros(STATE_DIM, dtype=np.float32)
    a = pol.predict_chunk(st, 0, 42)
    b = pol.predict_chunk(st, 0, 42)
    assert a.shape == (8, 14)
    assert float(np.max(np.abs(a - b))) < 1e-6
    c = pol.predict_chunk(st, 0, 43)
    assert float(np.mean(np.abs(a - c))) > 0.0
    assert policy_seed(0, 1, 2) != policy_seed(0, 1, 3)
    g = torch.Generator()
    g.manual_seed(0)
    x = ddim_sample(m, torch.zeros(1, STATE_DIM), torch.zeros(1, dtype=torch.long), sched, g, n_infer=10, eta=0.0)
    assert x.shape == (1, 8, 14)
