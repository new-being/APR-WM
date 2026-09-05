"""Train-only normalizer. Val/test use train stats. Binary process channels untouched."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .process_features import PROCESS_BINARY


@dataclass
class ChunkNormalizer:
    mu_s: np.ndarray
    sig_s: np.ndarray
    mu_a: np.ndarray
    sig_a: np.ndarray
    mu_p: np.ndarray
    sig_p: np.ndarray
    eps: float = 1e-6

    def hash(self) -> str:
        payload = np.concatenate(
            [self.mu_s.ravel(), self.sig_s.ravel(), self.mu_a.ravel(), self.sig_a.ravel(), self.mu_p.ravel(), self.sig_p.ravel()]
        )
        return hashlib.sha256(payload.tobytes()).hexdigest()

    @staticmethod
    def fit(states: np.ndarray, actions: np.ndarray, process: np.ndarray) -> "ChunkNormalizer":
        mu_s = states.mean(0)
        sig_s = states.std(0)
        mu_a = actions.reshape(-1, actions.shape[-1]).mean(0)
        sig_a = actions.reshape(-1, actions.shape[-1]).std(0)
        mu_p = process.mean(0)
        sig_p = process.std(0)
        for i, is_bin in enumerate(PROCESS_BINARY):
            if is_bin:
                mu_p[i] = 0.0
                sig_p[i] = 1.0
        return ChunkNormalizer(mu_s, np.maximum(sig_s, 1e-6), mu_a, np.maximum(sig_a, 1e-6), mu_p, np.maximum(sig_p, 1e-6))

    def n_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self.mu_s) / (self.sig_s + self.eps)

    def n_act(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mu_a) / (self.sig_a + self.eps)

    def n_proc(self, p: np.ndarray) -> np.ndarray:
        out = (p - self.mu_p) / (self.sig_p + self.eps)
        for i, is_bin in enumerate(PROCESS_BINARY):
            if is_bin:
                out[..., i] = p[..., i]
        return out

    def denorm_act(self, a: np.ndarray) -> np.ndarray:
        return a * (self.sig_a + self.eps) + self.mu_a

    def as_dict(self) -> dict:
        return {
            "mu_s": self.mu_s,
            "sig_s": self.sig_s,
            "mu_a": self.mu_a,
            "sig_a": self.sig_a,
            "mu_p": self.mu_p,
            "sig_p": self.sig_p,
        }

    @staticmethod
    def from_dict(d: dict) -> "ChunkNormalizer":
        return ChunkNormalizer(
            np.asarray(d["mu_s"]),
            np.asarray(d["sig_s"]),
            np.asarray(d["mu_a"]),
            np.asarray(d["sig_a"]),
            np.asarray(d["mu_p"]),
            np.asarray(d["sig_p"]),
        )
