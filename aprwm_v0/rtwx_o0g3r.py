"""RTWX-O0G3R: learned canonical correspondence (RGBD U-Net) + fixed Kabsch.

3 fresh seeds; L1 on x_O/D_O; no rotation loss; B0 direct RGB→R reference. O1 locked.
"""

from __future__ import annotations

import json
import os
import sys
import gc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import _e_R_deg, _quat_fix
from .rtwx_o0c import _R_to_quat, _geomedian_R
from .rtwx_o0d3 import _coord_net, _nchw
from .rtwx_o0g1b import _quat_to_R
from .rtwx_o0g2 import EPOCHS, RGB_SIZE, _iou, _nchw_rgb, _predict_masks, _train_unet
from .rtwx_o0g2r import RTWXO0G2RConfig, collect_split
from .rtwx_o0g3 import MAX_PAIRS, MED_ER_MAX, N_MIN, P90_ER_MAX, _kabsch
from .rtwx_o0v import CAM_HEAD, CAM_OBS, P_ANY_AGG, P_ANY_SEED
from .rtwx_x0c import FORMAL_N_STEPS, _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G3R_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g3r.learned_correspondence_kabsch.v2"
SEEDS = (29601, 29602, 29603)
N_TRAIN_EP = 24
N_VAL_EP = 12
N_TEST_EP = 12
JOINT_DET_MIN = 0.95
E_CORR_TRAIN_MAX = 0.05
MODE_RATIO = 0.5
CORR_EPOCHS = 25
CORR_LR = 1.0e-3
CORR_PATIENCE = 8
B0_EPOCHS = 40
B0_LR = 1.0e-3
B0_PATIENCE = 8
SPLIT_KEYS = (
    "p", "quat", "rgb_h", "rgb_o", "mask_h", "mask_o", "xyz_h", "xyz_o", "vis_h", "vis_o",
)


@dataclass(frozen=True)
class RTWXO0G3RConfig:
    output: str = "runs/rtwx_o0g3r"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    seeds: tuple[int, ...] = SEEDS
    n_train_ep: int = N_TRAIN_EP
    n_val_ep: int = N_VAL_EP
    n_test_ep: int = N_TEST_EP
    n_steps: int = FORMAL_N_STEPS
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    epochs_mask: int = EPOCHS
    epochs_corr: int = CORR_EPOCHS
    epochs_b0: int = B0_EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G3R must not write there")


def _lock(cfg: RTWXO0G3RConfig) -> RTWXO0G3RConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        seeds=SEEDS,
        n_train_ep=N_TRAIN_EP,
        n_val_ep=N_VAL_EP,
        n_test_ep=N_TEST_EP,
        n_steps=FORMAL_N_STEPS,
        rgb_size=RGB_SIZE,
        epochs_mask=EPOCHS,
        epochs_corr=CORR_EPOCHS,
        epochs_b0=B0_EPOCHS,
    )


def load_D_O(robotwin_repo: str | Path) -> float:
    path = Path(robotwin_repo) / "assets/objects/021_cup/model_data0.json"
    data = json.loads(path.read_text())
    ext = np.asarray(data["extents"], dtype=np.float64)
    scale = np.asarray(data["scale"], dtype=np.float64)
    return float(np.max(ext * scale))


def _xo_grid(xyz: np.ndarray, mask: np.ndarray, p: np.ndarray, quat: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(xyz).shape, np.nan, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & np.isfinite(xyz).all(axis=-1)
    if not m.any():
        return out
    R = _quat_to_R(quat)
    out[m] = ((xyz[m].astype(np.float64) - np.asarray(p, dtype=np.float64).reshape(1, 3)) @ R).astype(np.float32)
    return out


def _batch_xo(xyz: np.ndarray, mask: np.ndarray, p: np.ndarray, quat: np.ndarray) -> np.ndarray:
    n = xyz.shape[0]
    out = np.full((n, *xyz.shape[1:]), np.nan, dtype=np.float32)
    for i in range(n):
        out[i] = _xo_grid(xyz[i], mask[i], p[i], quat[i])
    return out


def _depth_map(xyz: np.ndarray) -> np.ndarray:
    z = np.asarray(xyz[..., 2], dtype=np.float32)
    return np.where(np.isfinite(z), z, 0.0)


def _rgbd_nchw(rgb: np.ndarray, xyz: np.ndarray, z_mu: float, z_sd: float) -> np.ndarray:
    rgb_f = np.asarray(rgb, dtype=np.float32) / 255.0
    dep = (_depth_map(xyz) - z_mu) / max(z_sd, 1e-6)
    dep = dep[..., None]
    x = np.concatenate([rgb_f, dep], axis=-1)
    return np.transpose(x, (0, 3, 1, 2))


def _small_unet_rgbd(n_out: int = 3, n_in: int = 4):
    from torch import nn
    import torch.nn.functional as F

    class ConvBN(nn.Module):
        def __init__(self, a: int, b: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e1 = ConvBN(n_in, 16)
            self.e2 = ConvBN(16, 32)
            self.e3 = ConvBN(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d2 = ConvBN(64, 32)
            self.u1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.d1 = ConvBN(32, 16)
            self.out = nn.Conv2d(16, n_out, 1)

        def forward(self, x):
            import torch
            x1 = self.e1(x)
            x2 = self.e2(self.pool(x1))
            x3 = self.e3(self.pool(x2))
            y2 = self.u2(x3)
            if y2.shape[-2:] != x2.shape[-2:]:
                y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            y2 = self.d2(torch.cat([y2, x2], dim=1))
            y1 = self.u1(y2)
            if y1.shape[-2:] != x1.shape[-2:]:
                y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
            y1 = self.d1(torch.cat([y1, x1], dim=1))
            return self.out(y1)

    return UNet()


def _z_stats(xyz: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(xyz[..., 2])
    z = xyz[..., 2][m]
    if z.size == 0:
        return 0.0, 1.0
    return float(z.mean()), float(max(z.std(), 1e-6))


def _train_corr(
    rgb, xyz, xo, mask, rgb_va, xyz_va, xo_va, mask_va, *, D_O: float, z_mu: float, z_sd: float, seed: int, epochs: int,
) -> Any:
    import torch

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _small_unet_rgbd(3, 4).to(dev)
    mtr, mva = np.asarray(mask, bool), np.asarray(mask_va, bool)
    xo_n = np.nan_to_num(xo, nan=0.0) / D_O
    xo_va_n = np.nan_to_num(xo_va, nan=0.0) / D_O
    xtr = torch.from_numpy(_rgbd_nchw(rgb, xyz, z_mu, z_sd))
    ytr = torch.from_numpy(xo_n.astype(np.float32)).permute(0, 3, 1, 2)
    wtr = torch.from_numpy(mtr.astype(np.float32)[:, None])
    xva = torch.from_numpy(_rgbd_nchw(rgb_va, xyz_va, z_mu, z_sd))
    yva = torch.from_numpy(xo_va_n.astype(np.float32)).permute(0, 3, 1, 2)
    wva = torch.from_numpy(mva.astype(np.float32)[:, None])
    opt = torch.optim.Adam(model.parameters(), lr=CORR_LR)
    bs, best, best_state, bad = 16, float("inf"), None, 0
    n = xtr.shape[0]
    for ep in range(epochs):
        model.train()
        for i in np.random.default_rng(seed + ep).permutation(n):
            xb, yb, wb = xtr[int(i) : int(i) + 1].to(dev), ytr[int(i) : int(i) + 1].to(dev), wtr[int(i) : int(i) + 1].to(dev)
            opt.zero_grad()
            pred = model(xb)
            loss = (torch.abs(pred - yb) * wb).sum() / wb.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for j in range(0, xva.shape[0], bs):
                pred = model(xva[j : j + bs].to(dev))
                wb = wva[j : j + bs].to(dev)
                vals.append(float((torch.abs(pred - yva[j : j + bs].to(dev)) * wb).sum() / wb.sum().clamp(min=1.0)))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= CORR_PATIENCE:
                break
        if ep % 5 == 0 or ep + 1 == epochs:
            print(f"[rtwx-o0g3r] corr ep={ep} val={v:.6f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    model._o0g3r_D_O = float(D_O)
    model._o0g3r_z_mu = float(z_mu)
    model._o0g3r_z_sd = float(z_sd)
    return model


def _predict_xo(model: Any, rgb: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    x = _rgbd_nchw(rgb, xyz, model._o0g3r_z_mu, model._o0g3r_z_sd)
    rows = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            rows.append(model(torch.from_numpy(x[i : i + 16]).to(dev)).cpu().numpy())
    y = np.concatenate(rows, axis=0)
    return np.transpose(y, (0, 2, 3, 1)) * model._o0g3r_D_O


def _train_b0(rgb: np.ndarray, quat: np.ndarray, rgb_va: np.ndarray, quat_va: np.ndarray, *, seed: int, epochs: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _coord_net(4, int(rgb.shape[1])).to(dev)
    mu = quat.mean(0)
    sd = quat.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ytr = ((quat - mu) / sd).astype(np.float32)
    yva = ((quat_va - mu) / sd).astype(np.float32)
    xtr = _nchw(rgb)
    xva = _nchw(rgb_va)
    opt = torch.optim.Adam(model.parameters(), lr=B0_LR)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(xtr.shape[0])
        for i in order:
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(xtr[int(i) : int(i) + 1]).to(dev)), torch.from_numpy(ytr[int(i) : int(i) + 1]).float().to(dev)).backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for j in range(0, xva.shape[0], 16):
                vals.append(float(loss_fn(model(torch.from_numpy(xva[j : j + 16]).to(dev)), torch.from_numpy(yva[j : j + 16]).float().to(dev))))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= B0_PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    model._b0_mu, model._b0_sd = mu, sd
    return model


def _predict_b0(model: Any, rgb: np.ndarray) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        for i in range(0, rgb.shape[0], 16):
            rows.append(model(torch.from_numpy(_nchw(rgb[i : i + 16])).to(dev)).cpu().numpy())
    y = np.concatenate(rows, axis=0)
    return _quat_fix(y * model._b0_sd + model._b0_mu)


def _subsample(X_O: np.ndarray, X_B: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if X_O.shape[0] <= MAX_PAIRS:
        return X_O, X_B
    idx = rng.choice(X_O.shape[0], MAX_PAIRS, replace=False)
    return X_O[idx], X_B[idx]


def _pairs_from_frame(mh, mo, xoh, xoo, xyzh, xyzo, uh, uo) -> tuple[np.ndarray, np.ndarray]:
    chunks_o, chunks_b = [], []
    for m, xo, xyz, u in ((mh, xoh, xyzh, uh), (mo, xoo, xyzo, uo)):
        sel = np.asarray(m, bool) & np.asarray(u, bool) & np.isfinite(xyz).all(-1) & np.isfinite(xo).all(-1)
        if sel.any():
            chunks_o.append(xo[sel].astype(np.float64))
            chunks_b.append(xyz[sel].astype(np.float64))
    if not chunks_o:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(chunks_o, 0), np.concatenate(chunks_b, 0)


def _kabsch_quat(mh, mo, xoh, xoo, xyzh, xyzo, uh, uo, rng) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    qh = qo = np.full(4, np.nan)
    X_O, X_B = _pairs_from_frame(mh, mo, xoh, xoo, xyzh, xyzo, uh, uo)
    if X_O.shape[0] < N_MIN:
        return np.full(4, np.nan), float("nan"), qh, qo
    X_O, X_B = _subsample(X_O, X_B, rng)
    R, t, rms = _kabsch(X_O, X_B)
    qf = _R_to_quat(R) if np.isfinite(R).all() else np.full(4, np.nan)
    for m, xo, xyz, u, slot in ((mh, xoh, xyzh, uh, "h"), (mo, xoo, xyzo, uo, "o")):
        sel = np.asarray(m, bool) & np.asarray(u, bool) & np.isfinite(xyz).all(-1) & np.isfinite(xo).all(-1)
        if sel.sum() < 3:
            continue
        Xo, Xb = xo[sel].astype(np.float64), xyz[sel].astype(np.float64)
        Xo, Xb = _subsample(Xo, Xb, rng)
        Ro, _, _ = _kabsch(Xo, Xb)
        if np.isfinite(Ro).all():
            if slot == "h":
                qh = _R_to_quat(Ro)
            else:
                qo = _R_to_quat(Ro)
    return qf, rms, qh, qo


def _corr_norm_err(xhat, xgt, mask, D_O) -> np.ndarray:
    m = np.asarray(mask, bool) & np.isfinite(xhat).all(-1) & np.isfinite(xgt).all(-1)
    if not m.any():
        return np.array([], dtype=np.float64)
    return np.linalg.norm(xhat[m] - xgt[m], axis=1) / D_O


def _score_ori(qhat: np.ndarray, qgt: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(qhat).all(1) & np.isfinite(qgt).all(1)
    if not ok.any():
        return {"median_e_R_deg": float("nan"), "p90_e_R_deg": float("nan"), "frac_near_90": float("nan"), "n_finite": 0}
    er = _e_R_deg(qhat[ok], qgt[ok])
    return {
        "median_e_R_deg": float(np.median(er)),
        "p90_e_R_deg": float(np.percentile(er, 90)),
        "frac_near_90": float(np.mean((er >= 75.0) & (er <= 105.0))),
        "n_finite": int(ok.sum()),
    }


def _eval_pool(
    pool: dict[str, Any], *, mask_model, corr_model, b0_model, D_O: float, use_learned_mask: bool, use_learned_xo: bool, use_b0: bool, rng: np.random.Generator,
) -> dict[str, Any]:
    n = pool["p"].shape[0]
    pred_h = pred_o = None
    if use_learned_mask:
        pred_h = _predict_masks(mask_model, pool["rgb_h"])
        pred_o = _predict_masks(mask_model, pool["rgb_o"])
    xo_gt_h = _batch_xo(pool["xyz_h"], pool["mask_h"], pool["p"], pool["quat"])
    xo_gt_o = _batch_xo(pool["xyz_o"], pool["mask_o"], pool["p"], pool["quat"])
    if use_learned_xo:
        xo_h = _predict_xo(corr_model, pool["rgb_h"], pool["xyz_h"])
        xo_o = _predict_xo(corr_model, pool["rgb_o"], pool["xyz_o"])
    else:
        xo_h, xo_o = xo_gt_h, xo_gt_o
    mh = pred_h if use_learned_mask else pool["mask_h"]
    mo = pred_o if use_learned_mask else pool["mask_o"]
    uh, uo = mh, mo
    qhat = np.full((n, 4), np.nan)
    rms = np.full(n, np.nan)
    dview = np.full(n, np.nan)
    corr_errs = []
    for i in range(n):
        if use_b0:
            qh = _predict_b0(b0_model, pool["rgb_h"][i : i + 1])[0]
            qo = _predict_b0(b0_model, pool["rgb_o"][i : i + 1])[0]
            Rs = []
            for q in (qh, qo):
                if np.isfinite(q).all():
                    Rs.append(_quat_to_R(q))
            qhat[i] = _R_to_quat(_geomedian_R(Rs)) if Rs else np.full(4, np.nan)
            rms[i] = float("nan")
            if len(Rs) == 2:
                dview[i] = float(_e_R_deg(qh.reshape(1, 4), qo.reshape(1, 4))[0])
        else:
            qhat[i], rms[i], qh, qo = _kabsch_quat(mh[i], mo[i], xo_h[i], xo_o[i], pool["xyz_h"][i], pool["xyz_o"][i], uh[i], uo[i], rng)
            if np.isfinite(qh).all() and np.isfinite(qo).all():
                dview[i] = float(_e_R_deg(qh.reshape(1, 4), qo.reshape(1, 4))[0])
        if use_learned_xo:
            corr_errs.append(np.concatenate([
                _corr_norm_err(xo_h[i], xo_gt_h[i], pool["mask_h"][i], D_O),
                _corr_norm_err(xo_o[i], xo_gt_o[i], pool["mask_o"][i], D_O),
            ]))
    ce = np.concatenate(corr_errs) if corr_errs else np.array([])
    ori = _score_ori(qhat, pool["quat"])
    return {
        **ori,
        "median_r_fit": float(np.nanmedian(rms)),
        "E_corr_norm_median": float(np.median(ce)) if ce.size else float("nan"),
        "E_corr_norm_p90": float(np.percentile(ce, 90)) if ce.size else float("nan"),
        "median_d_view": float(np.nanmedian(dview)),
    }


def _g2r_cfg(cfg: RTWXO0G3RConfig, seed: int) -> RTWXO0G2RConfig:
    return RTWXO0G2RConfig(
        output=cfg.output,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        n_train_ep=cfg.n_train_ep,
        n_val_ep=cfg.n_val_ep,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        seed=seed,
        seed_attempts=cfg.seed_attempts,
        max_resample=cfg.max_resample,
        smoke=cfg.smoke,
        rgb_size=cfg.rgb_size,
        epochs=cfg.epochs_mask,
    )


def _save_split(path: Path, name: str, pool: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": pool[k] for k in SPLIT_KEYS}
    payload[f"{name}_n_ep"] = np.array([pool["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([pool["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_split(path: Path, name: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in SPLIT_KEYS}
    p["rgb_h"] = p["rgb_h"].astype(np.uint8)
    p["rgb_o"] = p["rgb_o"].astype(np.uint8)
    for k in ("mask_h", "mask_o", "vis_h", "vis_o"):
        p[k] = p[k].astype(bool)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect_seed(cfg: RTWXO0G3RConfig, root: Path, seed: int, delta_O: np.ndarray, stop: list[str]) -> dict[str, dict[str, Any]]:
    g2r = _g2r_cfg(cfg, seed)
    out = {}
    rngs = {"train": np.random.default_rng(seed), "val": np.random.default_rng(seed + 1), "test": np.random.default_rng(seed + 2)}
    for split, rng in rngs.items():
        path = root / f"cache_o0g3r_s{seed}_{split}.npz"
        hit = _load_split(path, split)
        if hit is not None:
            print(f"[rtwx-o0g3r] seed={seed} {split}: cache", flush=True)
            out[split] = hit
            continue
        if cfg.backend == "numpy":
            from .rtwx_o0g2r import _numpy_split as _numpy_split_g2r
            pool = _numpy_split_g2r(g2r, split, rng, delta_O)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_raster_shader()
            _patch_curobo_planner(repo)
            pool = collect_split(g2r, split=split, rng=rng, stop=stop)
        out[split] = pool
        _save_split(path, split, pool)
    return out


def _concat_pools(pools: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: np.concatenate([p[k] for p in pools], axis=0) for k in SPLIT_KEYS}
    out["n_ep"] = sum(p["n_ep"] for p in pools)
    out["n_steps"] = pools[0]["n_steps"]
    return out


def _n_pairs_frame(mh, mo, xoh, xoo, xyzh, xyzo, uh, uo) -> int:
    return int(_pairs_from_frame(mh, mo, xoh, xoo, xyzh, xyzo, uh, uo)[0].shape[0])


def _g0_coverage(test: dict[str, Any], per_seed_pools: dict[int, dict[str, dict[str, Any]]], cfg: RTWXO0G3RConfig, pred_h, pred_o) -> tuple[bool, dict[str, Any]]:
    xo_h = _batch_xo(test["xyz_h"], test["mask_h"], test["p"], test["quat"])
    xo_o = _batch_xo(test["xyz_o"], test["mask_o"], test["p"], test["quat"])
    vis_any = test["vis_h"] | test["vis_o"]
    p_any = float(np.mean(vis_any))
    det_any = pred_h.any(axis=(1, 2)) | pred_o.any(axis=(1, 2))
    joint = float(np.mean(det_any[vis_any])) if vis_any.any() else 0.0
    npairs = np.array([
        _n_pairs_frame(test["mask_h"][i], test["mask_o"][i], xo_h[i], xo_o[i], test["xyz_h"][i], test["xyz_o"][i], test["mask_h"][i], test["mask_o"][i])
        for i in range(test["p"].shape[0])
    ])
    p_enough = float(np.mean((npairs >= N_MIN) & vis_any)) if vis_any.any() else 0.0
    per_seed_cov = []
    for seed in cfg.seeds:
        sp = per_seed_pools[int(seed)]["test"]
        va = sp["vis_h"] | sp["vis_o"]
        xoh = _batch_xo(sp["xyz_h"], sp["mask_h"], sp["p"], sp["quat"])
        xoo = _batch_xo(sp["xyz_o"], sp["mask_o"], sp["p"], sp["quat"])
        spairs = np.array([
            _n_pairs_frame(sp["mask_h"][i], sp["mask_o"][i], xoh[i], xoo[i], sp["xyz_h"][i], sp["xyz_o"][i], sp["mask_h"][i], sp["mask_o"][i])
            for i in range(sp["p"].shape[0])
        ])
        per_seed_cov.append({
            "seed": int(seed),
            "P_visible_any": float(np.mean(va)),
            "P_enough_pairs": float(np.mean((spairs >= N_MIN) & va)) if va.any() else 0.0,
            "ok": float(np.mean(va)) >= P_ANY_SEED,
        })
    g0 = bool(p_any >= P_ANY_AGG and joint >= JOINT_DET_MIN and p_enough >= 0.95 and all(r["ok"] for r in per_seed_cov))
    return g0, {
        "ok": g0,
        "P_visible_any": p_any,
        "joint_det": joint,
        "P_enough_pairs": p_enough,
        "N_min": N_MIN,
        "per_seed": per_seed_cov,
    }


def _release_cuda(tag: str = "rtwx-o0g3r") -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"[{tag}] cuda free={free / 1e9:.2f}GB / {total / 1e9:.2f}GB", flush=True)
    except Exception as exc:
        print(f"[{tag}] cuda reclaim skip: {exc}", flush=True)


def _pattern(*, g0: bool, g1: bool, g3: bool) -> str:
    if not g0:
        return "coverage_failure"
    if not g1:
        return "correspondence_instrument_failure"
    if not g3:
        return "geometry_mediated_orientation_failure"
    return "geometry_mediated_orientation_supported"


def run_rtwx_o0g3r(output: str | Path, config: RTWXO0G3RConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G3RConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    D_O = load_D_O(repo)
    from .rtwx_o0g1b import load_delta_O
    delta_O = np.asarray(load_delta_O(repo)["delta_O"], dtype=np.float64)

    header = {
        "stage": "RTWX-O0G3R",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "cameras": [CAM_HEAD, CAM_OBS],
        "method": "RGBD_UNet_x_O_Kabsch",
        "seeds": list(cfg.seeds),
        "D_O_m": D_O,
        "unlocks_o1": False,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    stop: list[str] = []
    per_seed_pools = {}
    for seed in cfg.seeds:
        print(f"[rtwx-o0g3r] === collect seed {seed} ===", flush=True)
        per_seed_pools[int(seed)] = _collect_seed(cfg, root, int(seed), delta_O, stop)
    if stop and cfg.backend != "numpy":
        raise RuntimeError(f"O0G3R collect stop: {stop}")

    try:
        os.chdir("/root/APR-WM")
    except OSError:
        pass
    _release_cuda()

    train = _concat_pools([per_seed_pools[s]["train"] for s in cfg.seeds])
    val = _concat_pools([per_seed_pools[s]["val"] for s in cfg.seeds])
    test = _concat_pools([per_seed_pools[s]["test"] for s in cfg.seeds])

    rgb_tr = np.concatenate([train["rgb_h"], train["rgb_o"]], 0)
    xyz_tr = np.concatenate([train["xyz_h"], train["xyz_o"]], 0)
    m_tr = np.concatenate([train["mask_h"], train["mask_o"]], 0)
    rgb_va = np.concatenate([val["rgb_h"], val["rgb_o"]], 0)
    xyz_va = np.concatenate([val["xyz_h"], val["xyz_o"]], 0)
    m_va = np.concatenate([val["mask_h"], val["mask_o"]], 0)
    z_mu, z_sd = _z_stats(xyz_tr, m_tr)

    ep_m = 4 if cfg.smoke else cfg.epochs_mask
    ep_c = 4 if cfg.smoke else cfg.epochs_corr
    ep_b0 = 4 if cfg.smoke else cfg.epochs_b0
    print(f"[rtwx-o0g3r] train mask U-Net epochs={ep_m}", flush=True)
    mask_model = _train_unet(rgb_tr, m_tr, rgb_va, m_va, seed=SEEDS[0], epochs=ep_m)
    _release_cuda()

    xo_tr = np.concatenate([
        _batch_xo(train["xyz_h"], train["mask_h"], train["p"], train["quat"]),
        _batch_xo(train["xyz_o"], train["mask_o"], train["p"], train["quat"]),
    ], 0)
    xo_va = np.concatenate([
        _batch_xo(val["xyz_h"], val["mask_h"], val["p"], val["quat"]),
        _batch_xo(val["xyz_o"], val["mask_o"], val["p"], val["quat"]),
    ], 0)
    print(f"[rtwx-o0g3r] train RGBD corr U-Net epochs={ep_c} D_O={D_O:.4f}", flush=True)
    corr_model = _train_corr(rgb_tr, xyz_tr, xo_tr, m_tr, rgb_va, xyz_va, xo_va, m_va, D_O=D_O, z_mu=z_mu, z_sd=z_sd, seed=SEEDS[0] + 11, epochs=ep_c)
    _release_cuda()

    # G1 instrument: train corr norm
    train_err = []
    for i in range(xo_tr.shape[0]):
        pred = _predict_xo(corr_model, rgb_tr[i : i + 1], xyz_tr[i : i + 1])[0]
        train_err.append(_corr_norm_err(pred, xo_tr[i], m_tr[i], D_O))
    e_train = float(np.median(np.concatenate(train_err))) if train_err else float("nan")
    g1 = bool(np.isfinite(e_train) and e_train < E_CORR_TRAIN_MAX)
    print(f"[rtwx-o0g3r] G1 instrument E_corr_train_norm={e_train:.4f} ok={g1}", flush=True)

    quat_tr = np.concatenate([train["quat"], train["quat"]], 0)
    quat_va = np.concatenate([val["quat"], val["quat"]], 0)
    print(f"[rtwx-o0g3r] train B0 direct RGB→R epochs={ep_b0}", flush=True)
    b0_model = _train_b0(rgb_tr, quat_tr, rgb_va, quat_va, seed=SEEDS[0] + 21, epochs=ep_b0)

    rng = np.random.default_rng(SEEDS[0] + 99)
    pred_h = _predict_masks(mask_model, test["rgb_h"])
    pred_o = _predict_masks(mask_model, test["rgb_o"])
    g0, g0_info = _g0_coverage(test, per_seed_pools, cfg, pred_h, pred_o)
    print(
        f"[rtwx-o0g3r] G0 P_any={g0_info['P_visible_any']:.3f} joint={g0_info['joint_det']:.3f} "
        f"P_enough={g0_info['P_enough_pairs']:.3f} ok={g0}",
        flush=True,
    )

    b0 = _eval_pool(test, mask_model=mask_model, corr_model=corr_model, b0_model=b0_model, D_O=D_O, use_learned_mask=False, use_learned_xo=False, use_b0=True, rng=rng)
    b1 = _eval_pool(test, mask_model=mask_model, corr_model=corr_model, b0_model=b0_model, D_O=D_O, use_learned_mask=False, use_learned_xo=False, use_b0=False, rng=rng)
    b2 = _eval_pool(test, mask_model=mask_model, corr_model=corr_model, b0_model=b0_model, D_O=D_O, use_learned_mask=True, use_learned_xo=True, use_b0=False, rng=rng)
    print(
        f"[rtwx-o0g3r] B0 f90={b0['frac_near_90']:.4f} B1 med={b1['median_e_R_deg']:.2f} "
        f"B2 med={b2['median_e_R_deg']:.2f} p90={b2['p90_e_R_deg']:.2f} f90={b2['frac_near_90']:.4f}",
        flush=True,
    )

    g3 = bool(np.isfinite(b2["median_e_R_deg"]) and b2["median_e_R_deg"] <= MED_ER_MAX and b2["p90_e_R_deg"] <= P90_ER_MAX)
    g4 = bool(np.isfinite(b2["frac_near_90"]) and np.isfinite(b0["frac_near_90"]) and b2["frac_near_90"] <= MODE_RATIO * b0["frac_near_90"])
    pattern = _pattern(g0=g0, g1=g1, g3=g3)
    mode_weak = bool(g3 and not g4)
    print(f"[rtwx-o0g3r] pattern={pattern} mode_reduction_weak={mode_weak}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "mode_reduction_weak": mode_weak,
        "G0_coverage": g0_info,
        "G1_instrument": {"ok": g1, "E_corr_train_norm": e_train, "gate": E_CORR_TRAIN_MAX},
        "G2_corr_descriptive": {"E_corr_norm_median": b2["E_corr_norm_median"], "E_corr_norm_p90": b2["E_corr_norm_p90"]},
        "G3_orientation": {"ok": g3, **{k: b2[k] for k in ("median_e_R_deg", "p90_e_R_deg", "frac_near_90", "n_finite")}},
        "G4_mode_reduction": {"ok": g4, "f90_B2": b2["frac_near_90"], "f90_B0": b0["frac_near_90"], "gate_ratio": MODE_RATIO},
        "G5_oracle_gap": {
            "delta_p90": float(b2["p90_e_R_deg"] - b1["p90_e_R_deg"]) if np.isfinite(b1["p90_e_R_deg"]) else float("nan"),
            "delta_median": float(b2["median_e_R_deg"] - b1["median_e_R_deg"]),
        },
        "B0_direct_RGB_to_R": b0,
        "B1_oracle_corr": b1,
        "B2_learned_primary": b2,
        "secondary": {"median_r_fit": b2["median_r_fit"], "median_d_view": b2["median_d_view"]},
        "localization": {
            "median_iou_head": float(np.median([_iou(pred_h[i], test["mask_h"][i]) for i in range(len(pred_h))])),
            "median_iou_observer": float(np.median([_iou(pred_o[i], test["mask_o"][i]) for i in range(len(pred_o))])),
        },
        "unlocks_o0c2_prereg": pattern == "geometry_mediated_orientation_supported",
        "unlocks_o1": False,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "G0_coverage", "G1_instrument", "G2_corr_descriptive", "G3_orientation", "G4_mode_reduction", "G5_oracle_gap", "B0_direct_RGB_to_R", "B1_oracle_corr", "B2_learned_primary", "unlocks_o0c2_prereg")})
    return summary
