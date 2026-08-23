"""PLAN-X1.5: learned action density vs mean+isotropic (no Hessian).

Iso / Diag / Mix-4. Does not revive H2. Does not trigger PLAN-X2.
Does not unlock R10. No diffusion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .capx_arm3_plant import HOST_PLANT_ID, SceneTheta, load_capx_arm3
from .plan_x0 import A_DIM, EPS_Q, U_ABS_MAX, rollout_cost
from .plan_x1 import (
    B_GRID,
    CEM_ITERS,
    CEM_TOTALS,
    ELITE_FRAC,
    SIGMA0,
    SIGMA_BROAD,
    auc_logb,
    bootstrap_mean_ci,
    cem_search,
    load_split,
)

PREREG_PATH = "REPORT/REG/PLANX/PLANX15_PREREG.md"
SCHEMA_ID = "aprwm.plan_x15.density.v1"
HIDDEN = 128
N_MIX = 4
LOG_SIG_MIN = float(np.log(0.02))
LOG_SIG_MAX = float(np.log(1.5))
TRAIN_SEEDS: tuple[int, ...] = (301, 302, 303, 304, 305)


@dataclass(frozen=True)
class PLANX15Config:
    x0_data: str = "runs/plan_x0/formal"
    output: str = "runs/plan_x15/formal"
    epochs: int = 40
    batch: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    patience: int = 8
    train_seeds: tuple[int, ...] = TRAIN_SEEDS
    max_eval_conditions: int | None = None
    skip_cem: bool = False
    physics_timestep: float = 0.002


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _success(q_t: np.ndarray, q_star: np.ndarray) -> bool:
    return float(np.max(np.abs(q_t - q_star))) < EPS_Q


def bound_log_sigma(raw):
    import torch

    return LOG_SIG_MIN + (LOG_SIG_MAX - LOG_SIG_MIN) * torch.sigmoid(raw)


def gaussian_nll(a, mu, log_sig):
    import torch

    z = (a - mu) / torch.exp(log_sig)
    return 0.5 * (z.square() + 2.0 * log_sig + np.log(2.0 * np.pi)).sum(dim=-1)


def mixture_nll(a, logits, mu, log_sig):
    """a: (B,D); logits: (B,M); mu/log_sig: (B,M,D)."""
    import torch
    from torch.nn import functional as F

    log_pi = F.log_softmax(logits, dim=-1)
    a_e = a.unsqueeze(1)
    nll_comp = gaussian_nll(a_e, mu, log_sig)
    return -torch.logsumexp(log_pi - nll_comp, dim=-1)


def build_iso(in_dim: int):
    import torch
    from torch import nn

    class IsoNet(nn.Module):
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

        def forward(self, x):
            return self.mu_head(self.backbone(x))

    return IsoNet()


def build_diag(in_dim: int):
    import torch
    from torch import nn

    class DiagNet(nn.Module):
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
            self.ls_head = nn.Linear(HIDDEN, A_DIM)

        def forward(self, x):
            z = self.backbone(x)
            return self.mu_head(z), bound_log_sigma(self.ls_head(z))

    return DiagNet()


def build_mix(in_dim: int, n_mix: int = N_MIX):
    import torch
    from torch import nn

    class MixNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_mix = n_mix
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, HIDDEN),
                nn.SiLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.SiLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.SiLU(),
            )
            self.pi_head = nn.Linear(HIDDEN, n_mix)
            self.mu_head = nn.Linear(HIDDEN, n_mix * A_DIM)
            self.ls_head = nn.Linear(HIDDEN, n_mix * A_DIM)
            nn.init.normal_(self.mu_head.weight, std=0.02)
            nn.init.uniform_(self.mu_head.bias, -0.15, 0.15)

        def forward(self, x):
            z = self.backbone(x)
            b = x.shape[0]
            logits = self.pi_head(z)
            mu = self.mu_head(z).view(b, self.n_mix, A_DIM)
            log_sig = bound_log_sigma(self.ls_head(z).view(b, self.n_mix, A_DIM))
            return logits, mu, log_sig

    return MixNet()


def _fit_loop(
    *,
    net,
    loss_fn: Callable,
    xt,
    at,
    xv,
    av,
    cfg: PLANX15Config,
    device,
    seed: int,
):
    import torch

    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(cfg.epochs):
        net.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for i0 in range(0, xt.shape[0], cfg.batch):
            idx = perm[i0 : i0 + cfg.batch]
            loss = loss_fn(net, xt[idx], at[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val_e = float(loss_fn(net, xv, av).mean().item())
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
    return net, float(best_val)


def train_family(
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    seed: int,
    cfg: PLANX15Config,
    device: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = np.stack([r["c"] for r in train]).astype(np.float32)
    a_tr = np.stack([r["A_star"] for r in train]).astype(np.float32)
    x_va = np.stack([r["c"] for r in val]).astype(np.float32)
    a_va = np.stack([r["A_star"] for r in val]).astype(np.float32)
    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1.0e-6
    xt = torch.from_numpy((x_tr - x_mean) / x_std).to(device)
    at = torch.from_numpy(a_tr).to(device)
    xv = torch.from_numpy((x_va - x_mean) / x_std).to(device)
    av = torch.from_numpy(a_va).to(device)
    in_dim = int(x_tr.shape[1])

    def iso_loss(net, x, a):
        import torch.nn.functional as F

        return F.mse_loss(net(x), a, reduction="none").mean(dim=-1)

    def diag_loss(net, x, a):
        mu, ls = net(x)
        return gaussian_nll(a, mu, ls)

    def mix_loss(net, x, a):
        logits, mu, ls = net(x)
        return mixture_nll(a, logits, mu, ls)

    iso, iso_v = _fit_loop(
        net=build_iso(in_dim).to(device),
        loss_fn=iso_loss,
        xt=xt,
        at=at,
        xv=xv,
        av=av,
        cfg=cfg,
        device=device,
        seed=seed,
    )
    diag, diag_v = _fit_loop(
        net=build_diag(in_dim).to(device),
        loss_fn=diag_loss,
        xt=xt,
        at=at,
        xv=xv,
        av=av,
        cfg=cfg,
        device=device,
        seed=seed + 17,
    )
    mix, mix_v = _fit_loop(
        net=build_mix(in_dim).to(device),
        loss_fn=mix_loss,
        xt=xt,
        at=at,
        xv=xv,
        av=av,
        cfg=cfg,
        device=device,
        seed=seed + 31,
    )

    def _x(c: np.ndarray):
        x = (np.asarray(c, dtype=np.float32).reshape(1, -1) - x_mean) / x_std
        return torch.from_numpy(x).to(device)

    def predict_iso(c: np.ndarray) -> np.ndarray:
        iso.eval()
        with torch.no_grad():
            return iso(_x(c)).cpu().numpy()[0].astype(np.float64)

    def predict_diag(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diag.eval()
        with torch.no_grad():
            mu, ls = diag(_x(c))
        return mu.cpu().numpy()[0].astype(np.float64), np.exp(ls.cpu().numpy()[0]).astype(np.float64)

    def predict_mix(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch.nn.functional as F

        mix.eval()
        with torch.no_grad():
            logits, mu, ls = mix(_x(c))
            pi = F.softmax(logits, dim=-1)
        return (
            pi.cpu().numpy()[0].astype(np.float64),
            mu.cpu().numpy()[0].astype(np.float64),
            np.exp(ls.cpu().numpy()[0]).astype(np.float64),
        )

    return {
        "seed": seed,
        "val": {"iso": iso_v, "diag": diag_v, "mix": mix_v},
        "predict_iso": predict_iso,
        "predict_diag": predict_diag,
        "predict_mix": predict_mix,
    }


def sample_mix(pi: np.ndarray, mu: np.ndarray, sig: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.choice(pi.shape[0], size=n, p=np.maximum(pi, 1.0e-12) / np.maximum(pi, 1.0e-12).sum())
    z = rng.normal(size=(n, A_DIM))
    return np.clip(mu[m] + sig[m] * z, -U_ABS_MAX, U_ABS_MAX)


def _roll_prefix(
    samples: np.ndarray,
    *,
    row: dict[str, Any],
    model: Any,
    data: Any,
    cache: dict[str, Any],
    theta: SceneTheta,
) -> tuple[np.ndarray, float]:
    succ = []
    ok = False
    j_mu = float("nan")
    for n_done, a in enumerate(samples, start=1):
        j, q_t, _ = rollout_cost(
            a,
            q0=row["q0"],
            qd0=row["qd0"],
            q_star=row["q_star"],
            theta=theta,
            model=model,
            data=data,
            cache=cache,
            u_abs_max=U_ABS_MAX,
        )
        if n_done == 1:
            j_mu = float(j)
        ok = ok or _success(q_t, row["q_star"])
        if n_done in B_GRID:
            succ.append(float(ok))
    return np.asarray(succ, dtype=np.float64), j_mu


def _pattern_full(g_h1: bool, g_iso: bool, g_mix: bool, mix_ci: dict[str, float], diag_ci: dict[str, float]) -> str:
    if not g_h1:
        return "h1_broken"
    if mix_ci["hi"] < 0:
        return "density_hurts"
    if g_mix:
        mix_vs_diag = bool(mix_ci["mean"] > diag_ci["mean"])
        diag_ok = bool(diag_ci["lo"] > 0)
        if diag_ok and not mix_vs_diag:
            return "diag_sufficient"
        return "density_beats_iso"
    return "iso_sufficient"


def run_plan_x15(
    output: str | Path | None = None,
    *,
    config: PLANX15Config | None = None,
) -> dict[str, Any]:
    cfg = config or PLANX15Config()
    root = Path(output or cfg.output)
    if "r10_c0" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X1.5 must not write under runs/r10_c0/")
    if "cap_x2" in str(root).replace("\\", "/"):
        raise RuntimeError("PLAN-X1.5 must not write under runs/cap_x2/")
    x0 = Path(cfg.x0_data)
    if not (x0 / "summary.json").is_file():
        raise RuntimeError(f"PLAN-X1.5 requires PLAN-X0 summary at {x0 / 'summary.json'}")
    x0s = json.loads((x0 / "summary.json").read_text(encoding="utf-8"))
    if not bool(x0s.get("plan_x0_passed")):
        raise RuntimeError("PLAN-X1.5 locked until plan_x0_passed")

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
        print(f"train seed={seed} (iso/diag/mix)", flush=True)
        bundles.append(train_family(train=train, val=val, seed=seed, cfg=cfg, device=device))

    model, data = load_capx_arm3(cfg.physics_timestep)
    n_test = len(test)
    names = ("broad", "iso", "diag", "mix")
    auc = {k: np.zeros(n_test) for k in names}
    s_grid = {k: np.zeros((n_test, len(B_GRID))) for k in names}
    j_iso = np.zeros(n_test)
    j_zero = np.zeros(n_test)
    cem_s = {int(b): {"van": np.zeros(n_test), "iso": np.zeros(n_test), "mix": np.zeros(n_test)} for b in CEM_TOTALS}

    n_max = max(B_GRID)
    for i, row in enumerate(test):
        if i % 25 == 0:
            print(f"eval {i}/{n_test}", flush=True)
        cache: dict[str, Any] = {}
        th = SceneTheta.from_vector(row["theta"])
        mus = [b["predict_iso"](row["c"]) for b in bundles]
        mu_iso = np.mean(mus, axis=0)
        diag_pack = [b["predict_diag"](row["c"]) for b in bundles]
        mu_d = np.mean([p[0] for p in diag_pack], axis=0)
        sig_d = np.exp(np.mean([np.log(np.maximum(p[1], 1.0e-8)) for p in diag_pack], axis=0))
        mix_pack = [b["predict_mix"](row["c"]) for b in bundles]

        j_iso[i], _, _ = rollout_cost(
            mu_iso, q0=row["q0"], qd0=row["qd0"], q_star=row["q_star"], theta=th, model=model, data=data, cache=cache, u_abs_max=U_ABS_MAX
        )
        j_zero[i], _, _ = rollout_cost(
            np.zeros(A_DIM), q0=row["q0"], qd0=row["qd0"], q_star=row["q_star"], theta=th, model=model, data=data, cache=cache, u_abs_max=U_ABS_MAX
        )

        rng = np.random.default_rng(30_000 + i)
        z = rng.normal(size=(n_max, A_DIM))
        samp_broad = np.clip(SIGMA_BROAD * z, -U_ABS_MAX, U_ABS_MAX)
        samp_iso = np.clip(mu_iso[None, :] + SIGMA0 * z, -U_ABS_MAX, U_ABS_MAX)
        samp_diag = np.clip(mu_d[None, :] + sig_d[None, :] * z, -U_ABS_MAX, U_ABS_MAX)
        # Equal mix of seed-wise mixture samples (do not average mixture components).
        n_per = int(np.ceil(n_max / len(bundles)))
        chunks = []
        for k, pack in enumerate(mix_pack):
            chunks.append(sample_mix(pack[0], pack[1], pack[2], n_per, np.random.default_rng(31_000 + i * 17 + k)))
        samp_mix = np.concatenate(chunks, axis=0)[:n_max]

        for name, samples in (("broad", samp_broad), ("iso", samp_iso), ("diag", samp_diag), ("mix", samp_mix)):
            succ, _ = _roll_prefix(samples, row=row, model=model, data=data, cache=cache, theta=th)
            s_grid[name][i] = succ
            auc[name][i] = auc_logb(B_GRID, succ)

        if not cfg.skip_cem:
            for tot in CEM_TOTALS:
                n_cand = tot // CEM_ITERS
                rng_c = np.random.default_rng(40_000 + i + tot)
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
                    sigma0=np.full(A_DIM, SIGMA_BROAD),
                )
                init_iso = np.clip(mu_iso[None, :] + SIGMA0 * rng_c.normal(size=(n_cand, A_DIM)), -U_ABS_MAX, U_ABS_MAX)
                _, ok_i = cem_search(
                    init_samples=init_iso,
                    n_cand=n_cand,
                    n_iters=CEM_ITERS,
                    row=row,
                    model=model,
                    data=data,
                    cache=cache,
                    rng=rng_c,
                    mu0=mu_iso,
                    sigma0=np.full(A_DIM, SIGMA0),
                )
                n_per_c = int(np.ceil(n_cand / len(mix_pack)))
                mix_chunks = []
                for k, pack in enumerate(mix_pack):
                    mix_chunks.append(sample_mix(pack[0], pack[1], pack[2], n_per_c, np.random.default_rng(41_000 + i + tot + k)))
                init_mix = np.concatenate(mix_chunks, axis=0)[:n_cand]
                _, ok_m = cem_search(
                    init_samples=init_mix,
                    n_cand=n_cand,
                    n_iters=CEM_ITERS,
                    row=row,
                    model=model,
                    data=data,
                    cache=cache,
                    rng=rng_c,
                    mu0=mu_iso,
                    sigma0=np.full(A_DIM, SIGMA0),
                )
                cem_s[tot]["van"][i] = float(ok_v)
                cem_s[tot]["iso"][i] = float(ok_i)
                cem_s[tot]["mix"][i] = float(ok_m)

    g_h1_ci = bootstrap_mean_ci(j_zero - j_iso, seed=1)
    g_h1 = bool(g_h1_ci["lo"] > 0)
    d_iso_broad = auc["iso"] - auc["broad"]
    g_iso_ci = bootstrap_mean_ci(d_iso_broad, seed=2)
    g_iso = bool(g_iso_ci["lo"] > 0)
    d_mix_iso = auc["mix"] - auc["iso"]
    g_mix_ci = bootstrap_mean_ci(d_mix_iso, seed=3)
    g_mix = bool(g_mix_ci["lo"] > 0)
    d_diag_iso = auc["diag"] - auc["iso"]
    diag_ci = bootstrap_mean_ci(d_diag_iso, seed=4)

    g_cem = False
    g_cem_ci = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    cem_auc = {}
    if not cfg.skip_cem:
        tot = CEM_TOTALS
        cem_auc_iso = []
        cem_auc_mix = []
        cem_auc_van = []
        for i in range(n_test):
            sv = np.array([cem_s[b]["van"][i] for b in tot])
            si = np.array([cem_s[b]["iso"][i] for b in tot])
            sm = np.array([cem_s[b]["mix"][i] for b in tot])
            cem_auc_van.append(auc_logb(tot, sv))
            cem_auc_iso.append(auc_logb(tot, si))
            cem_auc_mix.append(auc_logb(tot, sm))
        g_cem_ci = bootstrap_mean_ci(np.asarray(cem_auc_mix) - np.asarray(cem_auc_iso), seed=6)
        g_cem = bool(g_cem_ci["lo"] > 0)
        cem_auc = {
            "vanilla": float(np.mean(cem_auc_van)),
            "iso_seed": float(np.mean(cem_auc_iso)),
            "mix_seed": float(np.mean(cem_auc_mix)),
        }

    def s_mean(name: str) -> list[float]:
        return [float(np.mean(s_grid[name][:, k])) for k in range(len(B_GRID))]

    pattern = _pattern_full(g_h1, g_iso, g_mix, g_mix_ci, diag_ci)
    gates = {
        "G_H1": g_h1,
        "G_iso": g_iso,
        "G_mix": g_mix,
        "G_cem": g_cem if not cfg.skip_cem else None,
        "G_label": True,
    }
    passed = bool(g_h1 and g_iso and g_mix and gates["G_label"])
    summary = {
        "stage": "PLAN-X1.5",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host_plant_id": HOST_PLANT_ID,
        "unlocks_r10_c0": False,
        "diffusion": False,
        "uses_hessian": False,
        "x0_data": str(x0),
        "device": device,
        "n_eval": n_test,
        "config": asdict(cfg),
        "metrics": {
            "J_iso_mean": float(np.mean(j_iso)),
            "J_zero_mean": float(np.mean(j_zero)),
            "G_H1_deltaJ_ci": g_h1_ci,
            "AUC_S": {k: float(np.mean(v)) for k, v in auc.items()},
            "AUC_iso_minus_broad_ci": g_iso_ci,
            "AUC_mix_minus_iso_ci": g_mix_ci,
            "AUC_diag_minus_iso_ci": diag_ci,
            "S_vs_B": {k: s_mean(k) for k in names},
            "B_grid": list(B_GRID),
            "G_cem_ci": g_cem_ci,
            "CEM_AUC": cem_auc if not cfg.skip_cem else None,
            "CEM_S": {
                str(b): {
                    "vanilla": float(np.mean(cem_s[b]["van"])),
                    "iso_seed": float(np.mean(cem_s[b]["iso"])),
                    "mix_seed": float(np.mean(cem_s[b]["mix"])),
                }
                for b in CEM_TOTALS
            }
            if not cfg.skip_cem
            else None,
        },
        "gates": gates,
        "pattern": pattern,
        "plan_x15_passed": passed,
        "output": str(root.resolve()),
    }
    _write_json(root / "summary.json", summary)
    return summary
