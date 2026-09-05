"""AC1-D1 support geometries. Same whitening as AC0. k frozen. No learned OOD."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .state_support import K_NN, normalize_states

EPS = 1e-6
RIDGE = 1e-3
K_LOC = 30
D_EE_APPROACH = 0.20
PHASE_NAMES = ("approach", "engage", "transport", "terminate")


def phase_from_process(z: np.ndarray) -> int:
    d_ee, c_grasp, d_og, c_contact, c_goal = np.asarray(z, dtype=np.float64).reshape(-1)[:5]
    if c_goal >= 0.5:
        return 3
    if c_grasp >= 0.5 or c_contact >= 0.5:
        return 1
    if d_ee >= D_EE_APPROACH:
        return 0
    return 2


class EuclideanKNN:
    def __init__(self, train: np.ndarray, mu: np.ndarray, sig: np.ndarray, k: int = K_NN):
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sig = np.asarray(sig, dtype=np.float64).reshape(-1)
        tr = normalize_states(train, self.mu, self.sig)
        self.k_eff = min(int(k), max(int(tr.shape[0]), 1))
        self.tree = cKDTree(tr)

    def distances(self, states: np.ndarray) -> np.ndarray:
        x = normalize_states(states, self.mu, self.sig)
        d, _ = self.tree.query(x, k=self.k_eff)
        d = np.asarray(d, dtype=np.float64)
        return d if d.ndim == 1 else d.mean(axis=1)


class GlobalMahalanobis:
    def __init__(self, train: np.ndarray, mu: np.ndarray, sig: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sig = np.asarray(sig, dtype=np.float64).reshape(-1)
        tr = normalize_states(train, self.mu, self.sig)
        mean = tr.mean(0)
        dim = tr.shape[1]
        cov = np.cov(tr, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = np.asarray(cov, dtype=np.float64) + RIDGE * np.eye(dim)
        self.mean = mean
        self.prec = np.linalg.pinv(cov)

    def distances(self, states: np.ndarray) -> np.ndarray:
        x = normalize_states(states, self.mu, self.sig) - self.mean
        return np.sqrt(np.clip(np.sum((x @ self.prec) * x, axis=1), 0.0, None))


class PhaseConditionedKNN:
    def __init__(self, train: np.ndarray, phases: np.ndarray, mu: np.ndarray, sig: np.ndarray, k: int = K_NN):
        self.global_knn = EuclideanKNN(train, mu, sig, k=k)
        self.by_phase: dict[int, EuclideanKNN] = {}
        ph = np.asarray(phases, dtype=np.int64).reshape(-1)
        for p in range(4):
            idx = np.where(ph == p)[0]
            if idx.size >= k:
                self.by_phase[p] = EuclideanKNN(train[idx], mu, sig, k=k)

    def distances(self, states: np.ndarray, phases: np.ndarray) -> np.ndarray:
        st = np.asarray(states, dtype=np.float64)
        ph = np.asarray(phases, dtype=np.int64).reshape(-1)
        out = np.empty(st.shape[0], dtype=np.float64)
        for p in range(4):
            m = ph == p
            if not np.any(m):
                continue
            knn = self.by_phase.get(p, self.global_knn)
            out[m] = knn.distances(st[m])
        return out


class LocalDiagKNN:
    def __init__(self, train: np.ndarray, mu: np.ndarray, sig: np.ndarray, k_loc: int = K_LOC):
        self.mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        self.sig = np.asarray(sig, dtype=np.float64).reshape(-1)
        self.tr = normalize_states(train, self.mu, self.sig)
        self.k_loc = min(int(k_loc), max(int(self.tr.shape[0]), 1))
        self.tree = cKDTree(self.tr)

    def distances(self, states: np.ndarray) -> np.ndarray:
        x = normalize_states(states, self.mu, self.sig)
        _, idx = self.tree.query(x, k=self.k_loc)
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx.reshape(-1, 1)
        out = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            nb = self.tr[idx[i]]
            loc_mu = nb.mean(0)
            loc_sig = nb.std(0) + RIDGE
            out[i] = float(np.linalg.norm((x[i] - loc_mu) / loc_sig))
        return out
