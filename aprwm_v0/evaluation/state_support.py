"""Task-conditioned kNN state support. No learned OOD detector. k frozen."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K_NN = 5
ROBOT_SLICE = slice(0, 28)
OBJECT_SLICE = slice(28, 57)
EPS = 1e-6


def normalize_states(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mu) / (sig + EPS)


class TaskConditionedStateSupport:
    def __init__(self, train_states: np.ndarray, mu: np.ndarray, sig: np.ndarray, k: int = K_NN):
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sig = np.asarray(sig, dtype=np.float64).reshape(-1)
        self.k = int(k)
        tr = normalize_states(train_states, self.mu, self.sig)
        if tr.ndim != 2 or tr.shape[0] < 1:
            raise ValueError("train_states must be [N, D]")
        self.k_eff = min(self.k, int(tr.shape[0]))
        self.tree = cKDTree(tr)

    def distance(self, state: np.ndarray) -> float:
        x = normalize_states(np.asarray(state, dtype=np.float64).reshape(1, -1), self.mu, self.sig)
        d, _ = self.tree.query(x, k=self.k_eff)
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        return float(d.mean())

    def distances(self, states: np.ndarray) -> np.ndarray:
        x = normalize_states(states, self.mu, self.sig)
        d, _ = self.tree.query(x, k=self.k_eff)
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            return d
        return d.mean(axis=1)


def split_robot_object(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    return s[:, ROBOT_SLICE], s[:, OBJECT_SLICE]
