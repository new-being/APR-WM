"""PLAN-X1: unimodal sensitivity-aware Gaussian proposal (H1/H2).

Frozen: 3x128 SiLU, B0–B3, equal-trace Sigma, G0–G4. No diffusion.
Does not retune PLAN-X0. Does not reopen CAP-X2-P0. Does not unlock R10.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .capx_arm3_plant import HOST_PLANT_ID, N_DOF, SceneTheta, load_capx_arm3
from .plan_x0 import (
    A_DIM,
    EPS_Q,
    U_ABS_MAX,
    rollout_cost,
)

PREREG_PATH = "REPORT/REG/PLANX/PLANX1_PREREG.md"
SCHEMA_ID = "aprwm.plan_x1.proposal.v1"
HIDDEN = 128
SIGMA0 = 0.25
SIGMA_BROAD = 0.75
EPS_H_COV = 1.0e-6
H_FLOOR = 1.0e-8
LAMBDA_H = 1.0
TRAIN_SEEDS: tuple[int, ...] = (201, 202, 203, 204, 205)
B_GRID: tuple[int, ...] = (16, 32, 64, 128, 256)
CEM_TOTALS: tuple[int, ...] = (128, 256, 512, 1024)
CEM_ITERS = 4
ELITE_FRAC = 0.10
BOOTSTRAP_N = 2000


@dataclass(frozen=True)
class PLANX1Config:
    x0_data: str = "runs/plan_x0/formal"
    output: str = "runs/plan_x1/formal"
    epochs: int = 40
    batch: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    max_eval_conditions: int | None = None  # None = all test
    skip_cem: bool = False
    skip_cv: bool = False
    physics_timestep: float = 0.002


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_split(x0_root: Path, split: str) -> list[dict[str, Any]]:
    split_dir = x0_root / "data" / split
    rows: list[dict[str, Any]] = []
    for path in sorted(split_dir.glob("scene_*.h5")):
        with h5py.File(path, "r") as handle:
            theta = np.asarray(handle["theta"], dtype=np.float64)
            for key in handle.keys():
                if not key.startswith("cond_"):
                    continue
                g = handle[key]
                o = g["oracle"]
                q0 = np.asarray(g["q0"], dtype=np.float64)
                qd0 = np.asarray(g["qd0"], dtype=np.float64)
                q_star = np.asarray(g["q_star"], dtype=np.float64)
                c = np.concatenate([q0, qd0, q_star, theta]).astype(np.float32)
                rows.append(
                    {
                        "c": c,
                        "q0": q0,
                        "qd0": qd0,
                        "q_star": q_star,
                        "theta": theta,
                        "A_star": np.asarray(o["A_star"], dtype=np.float64),
                        "h": np.asarray(o["h"], dtype=np.float64),
                    }
                )
    return rows


def sigma_from_h(h: np.ndarray, *, sigma0: float = SIGMA0) -> np.ndarray:
    inv = 1.0 / (np.maximum(np.asarray(h, dtype=np.float64), H_FLOOR) + EPS_H_COV)
    scale = (A_DIM * sigma0**2) / float(np.sum(inv))
    return np.sqrt(scale * inv)


def auc_logb(b_grid: tuple[int, ...], s: np.ndarray) -> float:
    x = np.log(np.asarray(b_grid, dtype=np.float64))
    y = np.asarray(s, dtype=np.float64)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)) / (x[-1] - x[0] + 1.0e-12))


def bootstrap_mean_ci(deltas: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    means = np.empty(n, dtype=np.float64)
    for i in range(n):
        means[i] = float(np.mean(rng.choice(d, size=d.size, replace=True)))
    return {
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    if den < 1.0e-12:
        return float("nan")
    return float(np.sum(ra * rb) / den)


def build_net(in_dim: int = 22):
    import torch
    from torch import nn

    class Proposal(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, HIDDEN),
                nn.SiLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.SiLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.SiLU(),
            )
            self.mu_head = nn.Linear(HIDDEN, A_DIM)
            self.h_head = nn.Linear(HIDDEN, A_DIM)

        def forward(self, x):
            z = self.backbone(x)
            mu = self.mu_head(z)
            h = nn.functional.softplus(self.h_head(z)) + H_FLOOR
            return mu, h

    return Proposal()


def train_one(
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    seed: int,
    cfg: PLANX1Config,
    device: str,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = np.stack([r["c"] for r in train])
    a_tr = np.stack([r["A_star"] for r in train]).astype(np.float32)
    h_tr = np.log(np.maximum(np.stack([r["h"] for r in train]), H_FLOOR)).astype(np.float32)
    x_va = np.stack([r["c"] for r in val])
    a_va = np.stack([r["A_star"] for r in val]).astype(np.float32)
    h_va = np.log(np.maximum(np.stack([r["h"] for r in val]), H_FLOOR)).astype(np.float32)

    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1.0e-6
    x_trn = (x_tr - x_mean) / x_std
    x_van = (x_va - x_mean) / x_std

    dev = torch.device(device)
    net = build_net(x_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    xt = torch.from_numpy(x_trn).to(dev)
    at = torch.from_numpy(a_tr).to(dev)
    ht = torch.from_numpy(h_tr).to(dev)
    xv = torch.from_numpy(x_van).to(dev)
    av = torch.from_numpy(a_va).to(dev)
    hv = torch.from_numpy(h_va).to(dev)

    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=dev)
        for i0 in range(0, xt.shape[0], cfg.batch):
            idx = perm[i0 : i0 + cfg.batch]
            mu, hhat = net(xt[idx])
            loss = nn.functional.mse_loss(mu, at[idx]) + LAMBDA_H * nn.functional.huber_loss(
                torch.log(hhat), ht[idx]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            mu, hhat = net(xv)
            val_e = float(
                nn.functional.mse_loss(mu, av).item()
                + LAMBDA_H * nn.functional.huber_loss(torch.log(hhat), hv).item()
            )
        if val_e + 1.0e-12 < best_val:
            best_val = val_e
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    def predict_np(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        net.eval()
        x = (np.asarray(c, dtype=np.float32).reshape(1, -1) - x_mean) / x_std
        with torch.no_grad():
            mu, hhat = net(torch.from_numpy(x).to(dev))
        return mu.cpu().numpy()[0].astype(np.float64), hhat.cpu().numpy()[0].astype(np.float64)

    return {
        "seed": seed,
        "best_val": float(best_val),
        "predict": predict_np,
        "x_mean": x_mean,
        "x_std": x_std,
    }


def _success(q_t: np.ndarray, q_star: np.ndarray) -> bool:
    return float(np.max(np.abs(q_t - q_star))) < EPS_Q


def sample_and_roll(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    n: int,
    rng: np.random.Generator,
    row: dict[str, Any],
    model: Any,
    data: Any,
    cache: dict[str, Any],
) -> tuple[float, bool, np.ndarray]:
    """Return best J, any-success, best A among n samples."""

    best_j = float("inf")
    ok = False
    best_a = mu.copy()
    for _ in range(n):
        a = mu + sigma * rng.normal(size=A_DIM)
        a = np.clip(a, -U_ABS_MAX, U_ABS_MAX)
        j, q_t, _ = rollout_cost(
            a,
            q0=row["q0"],
            qd0=row["qd0"],
            q_star=row["q_star"],
            theta=SceneTheta.from_vector(row["theta"]),
            model=model,
            data=data,
            cache=cache,
            u_abs_max=U_ABS_MAX,
        )
        if j < best_j:
            best_j = j
            best_a = a
        ok = ok or _success(q_t, row["q_star"])
    return best_j, ok, best_a


def cv_delta_j(
    a_star: np.ndarray,
    sigma: np.ndarray,
    *,
    row: dict[str, Any],
    model: Any,
    data: Any,
    cache: dict[str, Any],
) -> float:
    j0, _, _ = rollout_cost(
        a_star,
        q0=row["q0"],
        qd0=row["qd0"],
        q_star=row["q_star"],
        theta=SceneTheta.from_vector(row["theta"]),
        model=model,
        data=data,
        cache=cache,
        u_abs_max=U_ABS_MAX,
    )
    djs = []
    for j in range(A_DIM):
        ap = a_star.copy()
        am = a_star.copy()
        ap[j] = np.clip(a_star[j] + sigma[j], -U_ABS_MAX, U_ABS_MAX)
        am[j] = np.clip(a_star[j] - sigma[j], -U_ABS_MAX, U_ABS_MAX)
        jp, _, _ = rollout_cost(
            ap,
            q0=row["q0"],
            qd0=row["qd0"],
            q_star=row["q_star"],
            theta=SceneTheta.from_vector(row["theta"]),
            model=model,
            data=data,
            cache=cache,
            u_abs_max=U_ABS_MAX,
        )
        jm, _, _ = rollout_cost(
            am,
            q0=row["q0"],
            qd0=row["qd0"],
            q_star=row["q_star"],
            theta=SceneTheta.from_vector(row["theta"]),
            model=model,
            data=data,
            cache=cache,
            u_abs_max=U_ABS_MAX,
        )
        djs.append(0.5 * (jp + jm) - j0)
    djs = np.asarray(djs, dtype=np.float64)
    mean = float(np.mean(djs))
    if abs(mean) < 1.0e-12:
        return float("nan")
    return float(np.std(djs) / mean)


def cem_search(
    *,
    init_samples: np.ndarray | None,
    n_cand: int,
    n_iters: int,
    row: dict[str, Any],
    model: Any,
    data: Any,
    cache: dict[str, Any],
    rng: np.random.Generator,
    mu0: np.ndarray,
    sigma0: np.ndarray,
) -> tuple[float, bool]:
    n_elite = max(1, int(round(ELITE_FRAC * n_cand)))
    mean = mu0.copy()
    std = np.maximum(sigma0.copy(), 1.0e-3)
    best_j = float("inf")
    best_ok = False
    for it in range(n_iters):
        if it == 0 and init_samples is not None:
            samples = init_samples
        else:
            samples = mean[None, :] + std[None, :] * rng.normal(size=(n_cand, A_DIM))
            samples = np.clip(samples, -U_ABS_MAX, U_ABS_MAX)
        js = np.zeros(n_cand)
        oks = np.zeros(n_cand, dtype=bool)
        for k in range(n_cand):
            j, q_t, _ = rollout_cost(
                samples[k],
                q0=row["q0"],
                qd0=row["qd0"],
                q_star=row["q_star"],
                theta=SceneTheta.from_vector(row["theta"]),
                model=model,
                data=data,
                cache=cache,
                u_abs_max=U_ABS_MAX,
            )
            js[k] = j
            oks[k] = _success(q_t, row["q_star"])
            if j < best_j:
                best_j = float(j)
                best_ok = bool(oks[k])
        elite = np.argpartition(js, n_elite - 1)[:n_elite]
        mean = samples[elite].mean(0)
        std = samples[elite].std(0) + 1.0e-3
    return best_j, best_ok


def _pattern(g0: bool, g3: bool, g4: bool, b3_vs_b1: bool, b2_vs_b1: bool) -> str:
    if not g0:
        return "proposal_failure"
    if g3 and b2_vs_b1:
        if g4:
            return "sensitivity_aware_proposal_success"
        return "sensitivity_sampling_success_cem_fail"
    if b3_vs_b1 and not b2_vs_b1:
        return "sensitivity_prediction_failure"
    if not b3_vs_b1 and not b2_vs_b1:
        return "anisotropy_no_value"
    return "mean_only_success"


def run_plan_x1(
    output: str | Path | None = None,
    *,
    config: PLANX1Config | None = None,
) -> dict[str, Any]:
    cfg = config or PLANX1Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X1 must not write under runs/r10_c0/")
    if "cap_x2" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X1 must not write under runs/cap_x2/")
    x0 = Path(cfg.x0_data)
    if not (x0 / "summary.json").is_file():
        raise RuntimeError(f"PLAN-X1 requires PLAN-X0 summary at {x0 / 'summary.json'}")
    x0s = json.loads((x0 / "summary.json").read_text(encoding="utf-8"))
    if not bool(x0s.get("plan_x0_passed")):
        raise RuntimeError("PLAN-X1 locked until plan_x0_passed")

    root.mkdir(parents=True, exist_ok=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading PLAN-X0 splits...", flush=True)
    train = load_split(x0, "train")
    val = load_split(x0, "val")
    test = load_split(x0, "test")
    if cfg.max_eval_conditions is not None:
        test = test[: cfg.max_eval_conditions]

    bundles = []
    for seed in cfg.train_seeds:
        print(f"train seed={seed}", flush=True)
        bundles.append(train_one(train=train, val=val, seed=seed, cfg=cfg, device=device))

    model, data = load_capx_arm3(cfg.physics_timestep)
    n_test = len(test)
    # Per-condition accumulators (mean over seeds where applicable).
    j_mu = np.zeros(n_test)
    j_zero = np.zeros(n_test)
    spear_s = []
    cv_sens = []
    cv_fix = []
    auc = {k: np.zeros(n_test) for k in ("B0", "B1", "B2", "B3")}
    s_grid = {k: np.zeros((n_test, len(B_GRID))) for k in ("B0", "B1", "B2", "B3")}
    cem_s = {int(b): {"van": np.zeros(n_test), "prop": np.zeros(n_test)} for b in CEM_TOTALS}

    for i, row in enumerate(test):
        if i % 25 == 0:
            print(f"eval {i}/{n_test}", flush=True)
        cache: dict[str, Any] = {}
        th = SceneTheta.from_vector(row["theta"])
        mus = []
        hs = []
        for b in bundles:
            mu, hh = b["predict"](row["c"])
            mus.append(mu)
            hs.append(hh)
        mu = np.mean(mus, axis=0)
        hhat = np.exp(np.mean(np.log(np.maximum(hs, H_FLOOR)), axis=0))
        sig_s = sigma_from_h(hhat)
        sig_o = sigma_from_h(row["h"])
        sig_f = np.full(A_DIM, SIGMA0)
        sig_b = np.full(A_DIM, SIGMA_BROAD)

        j_mu[i], _, _ = rollout_cost(
            mu, q0=row["q0"], qd0=row["qd0"], q_star=row["q_star"], theta=th, model=model, data=data, cache=cache, u_abs_max=U_ABS_MAX
        )
        j_zero[i], _, _ = rollout_cost(
            np.zeros(A_DIM), q0=row["q0"], qd0=row["qd0"], q_star=row["q_star"], theta=th, model=model, data=data, cache=cache, u_abs_max=U_ABS_MAX
        )
        pos = row["h"] > 0
        if np.any(pos):
            spear_s.append(spearman(np.log(hhat[pos]), np.log(np.maximum(row["h"][pos], H_FLOOR))))

        if not cfg.skip_cv:
            cv_sens.append(cv_delta_j(row["A_star"], sig_s, row=row, model=model, data=data, cache=cache))
            cv_fix.append(cv_delta_j(row["A_star"], sig_f, row=row, model=model, data=data, cache=cache))

        rng = np.random.default_rng(10_000 + i)
        specs = [("B0", np.zeros(A_DIM), sig_b), ("B1", mu, sig_f), ("B2", mu, sig_s), ("B3", mu, sig_o)]
        for name, m, sg in specs:
            succ = []
            # Incremental sampling: draw max B once, prefix.
            zs = rng.normal(size=(max(B_GRID), A_DIM))
            best_j = float("inf")
            ok = False
            for n_done, z in enumerate(zs, start=1):
                a = np.clip(m + sg * z, -U_ABS_MAX, U_ABS_MAX)
                j, q_t, _ = rollout_cost(
                    a, q0=row["q0"], qd0=row["qd0"], q_star=row["q_star"], theta=th, model=model, data=data, cache=cache, u_abs_max=U_ABS_MAX
                )
                best_j = min(best_j, j)
                ok = ok or _success(q_t, row["q_star"])
                if n_done in B_GRID:
                    succ.append(float(ok))
            s_grid[name][i] = np.asarray(succ)
            auc[name][i] = auc_logb(B_GRID, np.asarray(succ))

        if not cfg.skip_cem:
            for tot in CEM_TOTALS:
                n_cand = tot // CEM_ITERS
                rng_c = np.random.default_rng(20_000 + i + tot)
                _, ok_v = cem_search(
                    init_samples=None,
                    n_cand=n_cand,
                    n_iters=CEM_ITERS,
                    row=row,
                    model=model,
                    data=data,
                    cache=cache,
                    rng=rng_c,
                    mu0=np.zeros(A_DIM),
                    sigma0=sig_b,
                )
                init = np.clip(mu[None, :] + sig_s[None, :] * rng_c.normal(size=(n_cand, A_DIM)), -U_ABS_MAX, U_ABS_MAX)
                _, ok_p = cem_search(
                    init_samples=init,
                    n_cand=n_cand,
                    n_iters=CEM_ITERS,
                    row=row,
                    model=model,
                    data=data,
                    cache=cache,
                    rng=rng_c,
                    mu0=mu,
                    sigma0=sig_s,
                )
                cem_s[tot]["van"][i] = float(ok_v)
                cem_s[tot]["prop"][i] = float(ok_p)

    delta_j = j_zero - j_mu
    g0_ci = bootstrap_mean_ci(delta_j, seed=1)
    g0 = bool(g0_ci["lo"] > 0)

    # Spearman CI across conditions.
    sp = np.asarray(spear_s, dtype=np.float64)
    g1_ci = bootstrap_mean_ci(sp, seed=2)
    g1 = bool(g1_ci["mean"] > 0 and g1_ci["lo"] > 0)

    cv_s_m = float(np.nanmean(cv_sens)) if cv_sens else float("nan")
    cv_f_m = float(np.nanmean(cv_fix)) if cv_fix else float("nan")
    g2 = bool(np.isfinite(cv_s_m) and np.isfinite(cv_f_m) and cv_s_m <= 0.7 * cv_f_m)

    d21 = auc["B2"] - auc["B1"]
    g3_ci = bootstrap_mean_ci(d21, seed=3)
    g3 = bool(g3_ci["lo"] > 0)

    d31 = auc["B3"] - auc["B1"]
    b3_ci = bootstrap_mean_ci(d31, seed=4)
    b3_vs_b1 = bool(b3_ci["lo"] > 0)
    d10 = auc["B1"] - auc["B0"]
    b1_ci = bootstrap_mean_ci(d10, seed=5)

    g4 = False
    g4_ci = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    cem_auc_van = []
    cem_auc_prop = []
    if not cfg.skip_cem:
        # AUC over log B_total of success rates — per condition curve then delta.
        tot = CEM_TOTALS
        for i in range(n_test):
            sv = np.array([cem_s[b]["van"][i] for b in tot])
            spv = np.array([cem_s[b]["prop"][i] for b in tot])
            cem_auc_van.append(auc_logb(tot, sv))
            cem_auc_prop.append(auc_logb(tot, spv))
        g4_ci = bootstrap_mean_ci(np.asarray(cem_auc_prop) - np.asarray(cem_auc_van), seed=6)
        g4 = bool(g4_ci["lo"] > 0)

    def s_mean(name: str) -> list[float]:
        return [float(np.mean(s_grid[name][:, k])) for k in range(len(B_GRID))]

    pattern = _pattern(g0, g3, g4, b3_vs_b1, g3)
    gates = {
        "G0_mean": g0,
        "G1_spearman": g1,
        "G2_equal_cost": g2,
        "G3_auc_B2_gt_B1": g3,
        "G4_proposal_cem": g4 if not cfg.skip_cem else None,
        "G_label": True,
    }
    passed = bool(g0 and g1 and g2 and g3 and (g4 if not cfg.skip_cem else True) and gates["G_label"])
    summary = {
        "stage": "PLAN-X1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "unlocks_r10_c0": False,
        "diffusion": False,
        "x0_data": str(x0),
        "device": device,
        "n_eval": n_test,
        "config": asdict(cfg),
        "metrics": {
            "J_mu_mean": float(np.mean(j_mu)),
            "J_zero_mean": float(np.mean(j_zero)),
            "G0_deltaJ_ci": g0_ci,
            "spearman_ci": g1_ci,
            "CV_sens": cv_s_m,
            "CV_fixed": cv_f_m,
            "AUC_S": {k: float(np.mean(v)) for k, v in auc.items()},
            "AUC_B2_minus_B1_ci": g3_ci,
            "AUC_B3_minus_B1_ci": b3_ci,
            "AUC_B1_minus_B0_ci": b1_ci,
            "S_vs_B": {k: s_mean(k) for k in ("B0", "B1", "B2", "B3")},
            "B_grid": list(B_GRID),
            "G4_ci": g4_ci,
            "CEM_S": {
                str(b): {"vanilla": float(np.mean(cem_s[b]["van"])), "proposal": float(np.mean(cem_s[b]["prop"]))}
                for b in CEM_TOTALS
            }
            if not cfg.skip_cem
            else None,
        },
        "gates": gates,
        "pattern": pattern,
        "plan_x1_passed": passed,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
