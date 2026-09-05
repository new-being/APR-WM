"""Closed-loop causal history. Push after execute. Past actions = sent commands."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..data.history_windows import HS, PAST_A


class PolicyHistory:
    def reset(self, state0: np.ndarray, reset_action: np.ndarray) -> None:
        s0 = np.asarray(state0, dtype=np.float64).reshape(-1)
        a0 = np.asarray(reset_action, dtype=np.float64).reshape(-1)
        self.states = deque([s0.copy() for _ in range(HS)], maxlen=HS)
        self.actions = deque([a0.copy() for _ in range(PAST_A)], maxlen=PAST_A)

    def state_stack(self) -> np.ndarray:
        return np.stack(list(self.states), 0)

    def action_stack(self) -> np.ndarray:
        return np.stack(list(self.actions), 0)

    def update_after_step(self, new_state: np.ndarray, executed_command: np.ndarray) -> None:
        self.states.append(np.asarray(new_state, dtype=np.float64).reshape(-1).copy())
        self.actions.append(np.asarray(executed_command, dtype=np.float64).reshape(-1).copy())
