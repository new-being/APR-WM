"""Conditional diffusion chunk policy. 10 denoiser forwards = one planner call. No process/RGB/history."""

from __future__ import annotations

import numpy as np
import torch

from .deterministic_ddim import ddim_sample
from .diffusion_denoiser_mlp import DiffusionChunkDenoiser
from .diffusion_schedule import INFER_STEPS, TRAIN_STEPS, CosineSchedule


def policy_seed(task_id: int, episode_seed: int, tick: int) -> int:
    return int((int(episode_seed) * 100003 + int(task_id) * 10007 + int(tick) * 17) % (2**31 - 1))


class DiffusionChunkPolicy:
    fills_by_repeat = False
    fills_by_rollout = False
    uses_process = False
    uses_history = False
    uses_rgb = False
    planner_forwards_per_chunk = INFER_STEPS

    def __init__(self, denoiser: DiffusionChunkDenoiser, schedule: CosineSchedule, device: str, prediction_type: str = "epsilon"):
        self.denoiser = denoiser
        self.schedule = schedule
        self.device = device
        self.prediction_type = str(prediction_type)

    def eval(self) -> "DiffusionChunkPolicy":
        self.denoiser.eval()
        return self

    @torch.no_grad()
    def predict_chunk(
        self,
        state_n: np.ndarray,
        task_id: int,
        seed: int,
    ) -> np.ndarray:
        self.denoiser.eval()
        st = torch.from_numpy(np.asarray(state_n, dtype=np.float32).reshape(1, -1)).to(self.device)
        tid = torch.tensor([int(task_id)], dtype=torch.long, device=self.device)
        gen = torch.Generator()
        gen.manual_seed(int(seed) % (2**31 - 1))
        y = ddim_sample(
            self.denoiser, st, tid, self.schedule, gen, n_infer=INFER_STEPS, eta=0.0, prediction_type=self.prediction_type
        )
        return np.asarray(y[0].detach().cpu().numpy(), dtype=np.float64)
