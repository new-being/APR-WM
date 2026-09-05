"""RTWX-O0G5B: global-context CAD descriptor instrument (train memorization only).

No RANSAC. No continuous x^O. No fresh generalization. O0G5C/R/O1 locked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0g3_cache import xo_grid
from .rtwx_o0g3r import load_D_O
from .rtwx_o0g5a import SEEDS as G5A_SEEDS
from .rtwx_o0g5a import _load_split, load_cad_cloud
from .rtwx_o0v import CAM_HEAD, CAM_OBS
from .rtwx_x0c import _write_json

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0G5B_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0g5b.global_context_cad_descriptor.v1"
G5A_CACHE = "runs/rtwx_o0g5a"
K_ANCHOR = 512
N_SAMPLES = 8192
N_LOOKUP = 4096
N_GLOBAL = 128
PTS_PER_FRAME = 24
Z_DIM = 32
TAU = 0.07
E_MED_MAX = 0.10
TOPK_RECALL_MIN = 0.75
HIT_THRESH = 0.05
B0_ACC_MIN = 0.99
EPOCH_LUT = 800
EPOCH_DESC = 400
TOP_K = 5
FPS_SEED = 32601


@dataclass(frozen=True)
class RTWXO0G5BConfig:
    output: str = "runs/rtwx_o0g5b"
    g5a_cache: str = G5A_CACHE
    robotwin_repo: str = "/root/RoboTwin"
    seeds: tuple[int, ...] = G5A_SEEDS
    n_samples: int = N_SAMPLES
    n_lookup: int = N_LOOKUP
    k_anchor: int = K_ANCHOR
    epoch_lut: int = EPOCH_LUT
    epoch_desc: int = EPOCH_DESC
    smoke: bool = False


def _lock(cfg: RTWXO0G5BConfig) -> RTWXO0G5BConfig:
    if cfg.smoke:
        return replace(cfg, n_samples=1024, n_lookup=256, k_anchor=32, epoch_lut=80, epoch_desc=60)
    return replace(
        cfg,
        seeds=G5A_SEEDS,
        n_samples=N_SAMPLES,
        n_lookup=N_LOOKUP,
        k_anchor=K_ANCHOR,
        epoch_lut=EPOCH_LUT,
        epoch_desc=EPOCH_DESC,
    )


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0G5B must not write there")


def _cad_scale(robotwin_repo: str | Path) -> np.ndarray:
    import json

    path = Path(robotwin_repo) / "assets/objects/021_cup/model_data0.json"
    return np.asarray(json.loads(path.read_text())["scale"], dtype=np.float64)


def load_scaled_cad(robotwin_repo: str | Path, n_pts: int = 8192) -> np.ndarray:
    pts = load_cad_cloud(robotwin_repo, n_pts)
    return pts * _cad_scale(robotwin_repo).reshape(1, 3)


def _fps(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = pts.shape[0]
    k = min(k, n)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = int(rng.integers(0, n))
    dist = np.full(n, np.inf)
    for i in range(1, k):
        d = np.linalg.norm(pts - pts[idx[i - 1]], axis=1)
        dist = np.minimum(dist, d)
        idx[i] = int(np.argmax(dist))
    return pts[idx]


def _score(pred_anchor: np.ndarray, xo: np.ndarray, anchors: np.ndarray, scores: np.ndarray | None, D_O: float) -> dict[str, float]:
    err = np.linalg.norm(pred_anchor - xo, axis=1) / D_O
    y_gt = np.array([int(np.argmin(np.linalg.norm(anchors - xo[i], axis=1))) for i in range(xo.shape[0])])
    if scores is None:
        top5 = float(np.mean(err < HIT_THRESH))
        top1_acc = float("nan")
    else:
        top_idx = np.argsort(-scores, axis=1)[:, :TOP_K]
        hits = [float(np.min(np.linalg.norm(anchors[top_idx[i]] - xo[i], axis=1) / D_O) < HIT_THRESH) for i in range(xo.shape[0])]
        top5 = float(np.mean(hits))
        top1_acc = float(np.mean(np.argmax(scores, axis=1) == y_gt))
    return {
        "median_e_norm": float(np.median(err)),
        "p90_e_norm": float(np.percentile(err, 90)),
        "top1_label_acc": top1_acc,
        "top1_recall_r005": float(np.mean(err < HIT_THRESH)),
        "top5_recall": top5,
        "recall_r010": float(np.mean(err < 0.10)),
        "n": int(err.size),
    }


def _pack_instrument(pools: list[dict[str, Any]], *, D_O: float, rng: np.random.Generator, max_samples: int, k_global: int, per_frame: int) -> dict[str, np.ndarray]:
    loc, xo, gpts, gidx, area = [], [], [], [], []
    gi = 0
    for pool in pools:
        h, w = pool["rgb_h"].shape[1:3]
        for fi in range(pool["p"].shape[0]):
            for view in ("h", "o"):
                rgb = pool[f"rgb_{view}"][fi]
                xyz = pool[f"xyz_{view}"][fi]
                mask = np.asarray(pool[f"mask_{view}"][fi], bool)
                xog = xo_grid(xyz, mask, pool["p"][fi], pool["quat"][fi])
                m = mask & np.isfinite(xyz).all(-1) & np.isfinite(xog).all(-1)
                if m.sum() < 8:
                    continue
                ys, xs = np.where(m)
                cloud = xyz[m].astype(np.float64)
                cloud = cloud - cloud.mean(0, keepdims=True)
                if cloud.shape[0] > k_global:
                    pick = rng.choice(cloud.shape[0], k_global, replace=False)
                    gcloud = cloud[pick]
                else:
                    extra = rng.choice(cloud.shape[0], k_global - cloud.shape[0], replace=True)
                    gcloud = np.concatenate([cloud, cloud[extra]], 0)
                n_take = min(per_frame, ys.shape[0])
                take = rng.choice(ys.shape[0], n_take, replace=False)
                u = xs[take].astype(np.float32) / max(w - 1, 1)
                v = ys[take].astype(np.float32) / max(h - 1, 1)
                d = xyz[ys[take], xs[take], 2].astype(np.float32)
                col = rgb[ys[take], xs[take]].astype(np.float32) / 255.0
                loc.append(np.stack([u, v, d, col[:, 0], col[:, 1], col[:, 2]], axis=1))
                xo.append(xog[ys[take], xs[take]].astype(np.float64))
                gpts.append(gcloud.astype(np.float32))
                gidx.append(np.full(n_take, gi, dtype=np.int32))
                area.append(np.full(n_take, float(m.sum()), dtype=np.float32))
                gi += 1
    if not loc:
        raise RuntimeError("O0G5B: no instrument pixels")
    out = {
        "local": np.concatenate(loc, 0),
        "xo": np.concatenate(xo, 0),
        "global": np.stack(gpts, 0),
        "gidx": np.concatenate(gidx, 0),
        "area": np.concatenate(area, 0),
    }
    if out["local"].shape[0] > max_samples:
        idx = rng.choice(out["local"].shape[0], max_samples, replace=False)
        used = np.unique(out["gidx"][idx])
        remap = {int(o): n for n, o in enumerate(used)}
        out = {
            "local": out["local"][idx],
            "xo": out["xo"][idx],
            "global": out["global"][used],
            "gidx": np.array([remap[int(v)] for v in out["gidx"][idx]], dtype=np.int32),
            "area": out["area"][idx],
        }
    return out


def _labels(xo: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    # chunked NN
    y = np.empty(xo.shape[0], dtype=np.int64)
    for s in range(0, xo.shape[0], 1024):
        sl = slice(s, min(s + 1024, xo.shape[0]))
        d = ((xo[sl, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        y[sl] = np.argmin(d, axis=1)
    return y


def _train_lookup(y: np.ndarray, k: int, *, epochs: int, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    n = int(y.shape[0])
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LUT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e = nn.Embedding(n, 32)
            self.f = nn.Linear(32, k)

        def forward(self, i: torch.Tensor) -> torch.Tensor:
            return self.f(self.e(i))

    model = LUT().to(dev)
    idx = torch.arange(n, dtype=torch.long, device=dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(idx), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(idx).cpu().numpy()
    return logits


def _desc_models(*, local_only: bool, k: int, n_global: int):
    from torch import nn
    import torch

    class PointNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())

        def forward(self, cloud: torch.Tensor) -> torch.Tensor:
            return self.mlp(cloud).max(dim=1).values

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local_only = local_only
            self.loc = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.pn = None if local_only else PointNet()
            din = 32 if local_only else 96
            self.head = nn.Sequential(nn.Linear(din, 32), nn.ReLU(), nn.Linear(32, Z_DIM))
            self.cad = nn.Embedding(k, Z_DIM)

        def forward(self, loc: torch.Tensor, cloud: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.loc(loc)
            if self.local_only:
                z = self.head(h)
            else:
                g = self.pn(cloud)
                z = self.head(torch.cat([h, g], dim=1))
            z = nn.functional.normalize(z, dim=-1)
            cad = nn.functional.normalize(self.cad.weight, dim=-1)
            return z, cad

    return Net()


def _train_desc(
    local: np.ndarray,
    gidx: np.ndarray,
    glob: np.ndarray,
    y: np.ndarray,
    k: int,
    *,
    local_only: bool,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, np.ndarray]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _desc_models(local_only=local_only, k=k, n_global=glob.shape[1]).to(dev)
    loc_mu, loc_sd = local.mean(0), np.maximum(local.std(0), 1e-6)
    loc_n = ((local - loc_mu) / loc_sd).astype(np.float32)
    xt = torch.from_numpy(loc_n).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    gt = torch.from_numpy(glob.astype(np.float32)).to(dev)
    gi = torch.from_numpy(gidx.astype(np.int64)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xt.shape[0]
    bs = 256
    model.train()
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(n)
        for s in range(0, n, bs):
            b = torch.from_numpy(order[s : s + bs]).long().to(dev)
            opt.zero_grad()
            cloud = None if local_only else gt[gi[b]]
            z, cad = model(xt[b], cloud)
            logits = z @ cad.T / TAU
            nn.CrossEntropyLoss()(logits, yt[b]).backward()
            opt.step()
        if ep % 50 == 0 or ep + 1 == epochs:
            print(f"[rtwx-o0g5b] {'B1' if local_only else 'B2'} ep={ep}", flush=True)
    model.eval()
    with torch.no_grad():
        zs, cads = [], None
        for s in range(0, n, 512):
            cloud = None if local_only else gt[gi[s : s + 512]]
            z, cad = model(xt[s : s + 512], cloud)
            zs.append(z.cpu().numpy())
            cads = cad.cpu().numpy()
    return np.concatenate(zs, 0), cads, model, loc_mu, loc_sd


def _infer_desc(model: Any, local: np.ndarray, gidx: np.ndarray, glob: np.ndarray, *, local_only: bool, loc_mu: np.ndarray, loc_sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    loc_n = ((local - loc_mu) / loc_sd).astype(np.float32)
    xt = torch.from_numpy(loc_n).to(dev)
    gt = torch.from_numpy(glob.astype(np.float32)).to(dev)
    gi = torch.from_numpy(gidx.astype(np.int64)).to(dev)
    model.eval()
    zs, cads = [], None
    with torch.no_grad():
        for s in range(0, xt.shape[0], 512):
            cloud = None if local_only else gt[gi[s : s + 512]]
            z, cad = model(xt[s : s + 512], cloud)
            zs.append(z.cpu().numpy())
            cads = cad.cpu().numpy()
    return np.concatenate(zs, 0), cads


def _pattern(*, b0: bool, b2: bool) -> str:
    if not b0:
        return "target_pipeline_failure"
    if not b2:
        return "global_canonical_identity_not_representable"
    return "global_canonical_identity_supported"


def run_rtwx_o0g5b(output: str | Path, config: RTWXO0G5BConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0G5BConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    repo = cfg.robotwin_repo if Path(cfg.robotwin_repo).is_dir() else "/root/RoboTwin"
    D_O = load_D_O(repo)
    rng = np.random.default_rng(FPS_SEED)

    if cfg.smoke:
        from .rtwx_o0g5a import _numpy_cad_pool

        cad = _numpy_cad_pool(512, np.random.default_rng(0))
        n = cfg.n_samples
        n_g = max(n // 8, 1)
        samples = {
            "local": rng.normal(size=(n, 6)).astype(np.float32),
            "xo": cad[rng.integers(0, cad.shape[0], n)] + rng.normal(size=(n, 3)) * 0.002,
            "global": rng.normal(size=(n_g, 32, 3)).astype(np.float32),
            "gidx": np.clip(np.arange(n) // 8, 0, n_g - 1).astype(np.int32),
        }
    else:
        cad = load_scaled_cad(repo, 8192)
        pools = []
        for seed in cfg.seeds:
            path = Path(cfg.g5a_cache) / f"cache_o0g5a_s{seed}_test.npz"
            hit = _load_split(path, "test")
            if hit is None:
                raise FileNotFoundError(f"missing O0G5A cache {path}")
            pools.append(hit)
            print(f"[rtwx-o0g5b] seed={seed} cache", flush=True)
        samples = _pack_instrument(
            pools, D_O=D_O, rng=rng, max_samples=cfg.n_samples, k_global=N_GLOBAL, per_frame=PTS_PER_FRAME,
        )

    anchors = _fps(cad, cfg.k_anchor, np.random.default_rng(FPS_SEED))
    y = _labels(samples["xo"], anchors)
    quant = np.linalg.norm(anchors[y] - samples["xo"], axis=1) / D_O
    print(f"[rtwx-o0g5b] n={samples['xo'].shape[0]} K={anchors.shape[0]} quant_med={float(np.median(quant)):.4f}", flush=True)

    header = {
        "stage": "RTWX-O0G5B",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "no_ransac": True,
        "cameras": [CAM_HEAD, CAM_OBS],
        "K_anchor": int(anchors.shape[0]),
        "D_O_m": D_O,
        "quantization_median": float(np.median(quant)),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds)},
    }
    _write_json(root / "header.json", header)

    n_lut = min(cfg.n_lookup, y.shape[0])
    lut_idx = rng.choice(y.shape[0], n_lut, replace=False)
    print(f"[rtwx-o0g5b] B0 lookup n={n_lut}", flush=True)
    logits0 = _train_lookup(y[lut_idx], anchors.shape[0], epochs=cfg.epoch_lut, seed=FPS_SEED)
    pred0 = np.argmax(logits0, axis=1)
    acc0 = float(np.mean(pred0 == y[lut_idx]))
    sc0 = _score(anchors[pred0], samples["xo"][lut_idx], anchors, logits0, D_O)
    b0_ok = bool(acc0 >= B0_ACC_MIN)
    print(f"[rtwx-o0g5b] B0 acc={acc0:.4f} med={sc0['median_e_norm']:.4f} ok={b0_ok}", flush=True)

    print("[rtwx-o0g5b] B1 local descriptor", flush=True)
    z1, cad1, _, _, _ = _train_desc(
        samples["local"], samples["gidx"], samples["global"], y, anchors.shape[0],
        local_only=True, epochs=cfg.epoch_desc, seed=FPS_SEED + 1,
    )
    logits1 = z1 @ cad1.T
    pred1 = np.argmax(logits1, axis=1)
    sc1 = _score(anchors[pred1], samples["xo"], anchors, logits1, D_O)
    b1_ok = bool(sc1["median_e_norm"] <= E_MED_MAX and sc1["top5_recall"] >= TOPK_RECALL_MIN)
    print(f"[rtwx-o0g5b] B1 med={sc1['median_e_norm']:.4f} top5={sc1['top5_recall']:.3f} ok={b1_ok}", flush=True)

    print("[rtwx-o0g5b] B2 global-context descriptor", flush=True)
    z2, cad2, _, _, _ = _train_desc(
        samples["local"], samples["gidx"], samples["global"], y, anchors.shape[0],
        local_only=False, epochs=cfg.epoch_desc, seed=FPS_SEED + 2,
    )
    logits2 = z2 @ cad2.T
    pred2 = np.argmax(logits2, axis=1)
    sc2 = _score(anchors[pred2], samples["xo"], anchors, logits2, D_O)
    b2_ok = bool(sc2["median_e_norm"] <= E_MED_MAX and sc2["top5_recall"] >= TOPK_RECALL_MIN)
    print(f"[rtwx-o0g5b] B2 med={sc2['median_e_norm']:.4f} top5={sc2['top5_recall']:.3f} ok={b2_ok}", flush=True)

    pattern = _pattern(b0=b0_ok, b2=b2_ok)
    print(f"[rtwx-o0g5b] pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "B0_lookup": {"ok": b0_ok, "label_acc": acc0, **sc0, "gate_acc": B0_ACC_MIN},
        "B1_local": {"ok": b1_ok, **sc1},
        "B2_global": {"ok": b2_ok, **sc2, "gate": {"med": E_MED_MAX, "top5": TOPK_RECALL_MIN}},
        "quantization_ceiling": {"median_e_norm": float(np.median(quant)), "p90_e_norm": float(np.percentile(quant, 90))},
        "unlocks_o0g5c_prereg": pattern == "global_canonical_identity_supported",
        "unlocks_o0g5r": False,
        "unlocks_o0c2": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary
