"""RTWX-O0G3B: canonical correspondence instrument closure (train memorization only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g3_cache import N_MIN, SEEDS, load_D_O, load_g3r_pools, pack_pixel_samples, xo_grid
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G3B_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g3b.correspondence_instrument_closure.v1"
G3R_CACHE = "runs/rtwx_o0g3r"
E_CORR_MAX = 0.05
N_SAMPLES = 16384
N_LOOKUP = 4096
CROP_SIZE = 32
EPOCH_LUT = 800
EPOCH_MLP = 800
EPOCH_CROP = 400
PHI_TAU = 0.08
XO_TAU = 0.15
AMBIG_FRAC = 0.10


@dataclass(frozen=True)
class RTWXO0G3BConfig:
    output: str = "runs/rtwx_o0g3b"
    g3r_cache: str = G3R_CACHE
    robotwin_repo: str = "/root/RoboTwin"
    seeds: tuple[int, ...] = SEEDS
    n_samples: int = N_SAMPLES
    n_lookup: int = N_LOOKUP
    e_corr_max: float = E_CORR_MAX
    epoch_lut: int = EPOCH_LUT
    epoch_mlp: int = EPOCH_MLP
    epoch_crop: int = EPOCH_CROP
    smoke: bool = False


def _lock(cfg: RTWXO0G3BConfig) -> RTWXO0G3BConfig:
    if cfg.smoke:
        return replace(cfg, n_samples=2048, n_lookup=512, epoch_lut=200, epoch_mlp=200, epoch_crop=100)
    return replace(cfg, seeds=SEEDS, n_samples=N_SAMPLES, n_lookup=N_LOOKUP, epoch_lut=EPOCH_LUT, epoch_mlp=EPOCH_MLP, epoch_crop=EPOCH_CROP)


def _e_corr_norm(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(pred - tgt, axis=1)))


def _fit_torch(model: Any, x: np.ndarray, y: np.ndarray, *, epochs: int, seed: int, lr: float = 1e-3) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(xt).detach().cpu().numpy()


def _lookup_3d(y: np.ndarray, *, epochs: int, seed: int) -> tuple[np.ndarray, float]:
    import torch
    from torch import nn

    n = int(y.shape[0])
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LUT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e = nn.Embedding(n, 32)
            self.f = nn.Linear(32, 3)

        def forward(self, i: torch.Tensor) -> torch.Tensor:
            return self.f(self.e(i))

    model = LUT().to(dev)
    idx = torch.arange(n, dtype=torch.long, device=dev)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        nn.L1Loss()(model(idx), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        hat = model(idx).cpu().numpy()
    return hat, _e_corr_norm(hat, y)


def _mlp_uvd(uvd: np.ndarray, y: np.ndarray, *, epochs: int, seed: int) -> tuple[np.ndarray, float]:
    from torch import nn

    mu, sd = uvd.mean(0), uvd.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    x = ((uvd - mu) / sd).astype(np.float32)
    model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    hat = _fit_torch(model, x, y.astype(np.float32), epochs=epochs, seed=seed, lr=1e-3)
    return hat, _e_corr_norm(hat, y)


def _crop_samples(pool: dict[str, Any], *, D_O: float, rng: np.random.Generator, n_frames: int, crop_size: int, epochs: int) -> tuple[np.ndarray, float]:
    import torch
    from torch import nn

    xs, ys = [], []
    n = pool["p"].shape[0]
    picks = rng.choice(n, size=min(n_frames, n), replace=False)
    for fi in picks:
        for view in ("h", "o"):
            rgb = pool[f"rgb_{view}"][fi]
            xyz = pool[f"xyz_{view}"][fi]
            mask = np.asarray(pool[f"mask_{view}"][fi], bool)
            xo = xo_grid(xyz, mask, pool["p"][fi], pool["quat"][fi]) / D_O
            m = mask & np.isfinite(xyz).all(-1) & np.isfinite(xo).all(-1)
            if not m.any():
                continue
            ys_idx, xs_idx = np.where(m)
            y0, y1 = max(0, ys_idx.min() - 2), min(rgb.shape[0], ys_idx.max() + 3)
            x0, x1 = max(0, xs_idx.min() - 2), min(rgb.shape[1], xs_idx.max() + 3)
            crop_rgb = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
            crop_d = np.where(np.isfinite(xyz[y0:y1, x0:x1, 2]), xyz[y0:y1, x0:x1, 2], 0.0)
            crop_xo = np.nan_to_num(xo[y0:y1, x0:x1], nan=0.0)
            crop_m = m[y0:y1, x0:x1]
            # resize via simple nearest on grid
            ch, cw = crop_rgb.shape[:2]
            gy = np.linspace(0, ch - 1, crop_size).astype(int)
            gx = np.linspace(0, cw - 1, crop_size).astype(int)
            rr = crop_rgb[gy][:, gx]
            dd = crop_d[gy][:, gx]
            tt = crop_xo[gy][:, gx]
            mm = crop_m[gy][:, gx]
            feat = np.concatenate([rr, dd[..., None]], axis=-1)
            xs.append(feat[mm].reshape(-1, 4))
            ys.append(tt[mm])
    if not xs:
        raise RuntimeError("crop diagnostic: no crops")
    x_all = np.concatenate(xs, 0)
    y_all = np.concatenate(ys, 0)
    if x_all.shape[0] > 8192:
        idx = rng.choice(x_all.shape[0], 8192, replace=False)
        x_all, y_all = x_all[idx], y_all[idx]
    model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3))
    hat = _fit_torch(model, x_all, y_all.astype(np.float32), epochs=epochs, seed=42, lr=1e-3)
    return hat, _e_corr_norm(hat, y_all)


def _identifiability_audit(rgb: np.ndarray, uvd: np.ndarray, xo: np.ndarray, *, rng: np.random.Generator, n_audit: int = 2048) -> dict[str, float]:
    n = rgb.shape[0]
    if n > n_audit:
        idx = rng.choice(n, n_audit, replace=False)
        rgb, uvd, xo = rgb[idx], uvd[idx], xo[idx]
        n = n_audit
    phi = np.concatenate([rgb, (uvd - uvd.mean(0)) / np.maximum(uvd.std(0), 1e-6)], axis=1)
    # subsample pairs
    n_pairs = min(5000, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    dphi = np.linalg.norm(phi[i_idx] - phi[j_idx], axis=1)
    dxo = np.linalg.norm(xo[i_idx] - xo[j_idx], axis=1)
    ambig = (dphi < PHI_TAU) & (dxo > XO_TAU)
    return {
        "n_pairs": int(len(dphi)),
        "frac_ambiguous": float(np.mean(ambig)) if ambig.size else float("nan"),
        "median_dphi_near": float(np.median(dphi[ambig])) if ambig.any() else float("nan"),
        "median_dxo_near": float(np.median(dxo[ambig])) if ambig.any() else float("nan"),
        "phi_tau": PHI_TAU,
        "xo_tau": XO_TAU,
    }


def _pattern(*, b0: bool, b1: bool, b2: bool, ambig: bool) -> str:
    if not b0:
        return "target_pipeline_failure"
    if ambig:
        return "canonical_correspondence_ambiguous"
    if not b1:
        return "pixel_coord_not_representable"
    if not b2:
        return "crop_rgbd_not_representable"
    return "correspondence_instrument_closed"


def run_rtwx_o0g3b(output: str | Path, config: RTWXO0G3BConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G3BConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    D_O = load_D_O(cfg.robotwin_repo)
    rng = np.random.default_rng(30601)
    pool = load_g3r_pools(cfg.g3r_cache, cfg.seeds, "train")
    samples = pack_pixel_samples(pool, D_O=D_O, rng=rng, max_samples=cfg.n_samples)

    header = {
        "stage": "RTWX-O0G3B",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "depends_on": "O0G3R=coverage_failure",
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    n_lut = min(cfg.n_lookup, samples["xo"].shape[0])
    lut_idx = rng.choice(samples["xo"].shape[0], n_lut, replace=False)
    y_lut = samples["xo"][lut_idx]
    print(f"[rtwx-o0g3b] B0 lookup n={n_lut}", flush=True)
    hat_lut, e_lut = _lookup_3d(y_lut, epochs=cfg.epoch_lut if not cfg.smoke else 50, seed=30601)
    b0_ok = bool(e_lut < cfg.e_corr_max)
    print(f"[rtwx-o0g3b] B0 E_corr_norm={e_lut:.4f} ok={b0_ok}", flush=True)

    print("[rtwx-o0g3b] B1 (u,v,d)->xO MLP", flush=True)
    hat_mlp, e_mlp = _mlp_uvd(samples["uvd"], samples["xo"], epochs=cfg.epoch_mlp if not cfg.smoke else 50, seed=30602)
    b1_ok = bool(e_mlp < cfg.e_corr_max)
    print(f"[rtwx-o0g3b] B1 E_corr_norm={e_mlp:.4f} ok={b1_ok}", flush=True)

    print("[rtwx-o0g3b] B2 oracle-mask crop RGBD", flush=True)
    _, e_crop = _crop_samples(
        pool, D_O=D_O, rng=rng, n_frames=48 if not cfg.smoke else 8, crop_size=CROP_SIZE,
        epochs=cfg.epoch_crop if not cfg.smoke else 50,
    )
    b2_ok = bool(e_crop < cfg.e_corr_max)
    print(f"[rtwx-o0g3b] B2 E_corr_norm={e_crop:.4f} ok={b2_ok}", flush=True)

    audit = _identifiability_audit(samples["rgb"], samples["uvd"], samples["xo"], rng=rng)
    ambig = bool(np.isfinite(audit["frac_ambiguous"]) and audit["frac_ambiguous"] >= AMBIG_FRAC)
    print(f"[rtwx-o0g3b] identifiability frac_ambig={audit['frac_ambiguous']:.4f}", flush=True)

    pattern = _pattern(b0=b0_ok, b1=b1_ok, b2=b2_ok, ambig=ambig)
    print(f"[rtwx-o0g3b] pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "D_O_m": D_O,
        "n_train_pixels": int(samples["xo"].shape[0]),
        "B0_lookup": {"E_corr_norm": e_lut, "ok": b0_ok, "gate": cfg.e_corr_max},
        "B1_uvd_mlp": {"E_corr_norm": e_mlp, "ok": b1_ok, "gate": cfg.e_corr_max},
        "B2_crop_rgbd": {"E_corr_norm": e_crop, "ok": b2_ok, "gate": cfg.e_corr_max},
        "identifiability_audit": audit,
        "unlocks_o0g3r2_prereg": pattern == "correspondence_instrument_closed",
        "unlocks_o0c2": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary
