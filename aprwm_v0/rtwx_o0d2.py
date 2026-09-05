"""RTWX-O0D2: perception instrument closure. No O0R. No scientific claim."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args
from .rtwx_o0d import _blob_frame, _load_o0_train
from .rtwx_o0d1 import _project
from .rtwx_x0 import _nrmse
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader, _resize_rgb

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0D2_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0d2.perception_instrument_closure.v1"
SEED = 19601
N_EP = 6
N_STEPS = 20
N_MEM = 32
LUT_MAX = 1.0e-3
PIX_MAX = 0.05
FOV_MIN = 0.90
EPOCH_LUT = 1500
EPOCH_MLP = 1500
EPOCH_CNN = 600


@dataclass(frozen=True)
class RTWXO0D2Config:
    output: str = "runs/rtwx_o0d2"
    o0_cache: str = "runs/rtwx_o0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = N_EP
    n_steps: int = N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    n_mem: int = N_MEM
    epoch_lut: int = EPOCH_LUT
    epoch_mlp: int = EPOCH_MLP
    epoch_cnn: int = EPOCH_CNN


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0D2 must not write there")


def _lock(cfg: RTWXO0D2Config) -> RTWXO0D2Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=N_EP,
        n_steps=N_STEPS,
        seed=SEED,
        n_mem=N_MEM,
        epoch_lut=EPOCH_LUT,
        epoch_mlp=EPOCH_MLP,
        epoch_cnn=EPOCH_CNN,
    )


def _std_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu, sd = float(y.mean()), float(y.std())
    if sd < 1e-8:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def _fit_torch(model: Any, x: Any, y_std: np.ndarray, *, epochs: int, seed: int, lr: float = 1.0e-3) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    t = torch.from_numpy(np.asarray(y_std, dtype=np.float32).reshape(-1, 1)).to(dev)
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
    else:
        x = x.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x), t).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(x).detach().cpu().numpy().reshape(-1)


def _nrmse_denorm(hat_z: np.ndarray, y: np.ndarray, mu: float, sd: float) -> float:
    return _nrmse(hat_z * sd + mu, np.asarray(y, dtype=np.float64).reshape(-1))


def _lookup_nrmse(y: np.ndarray, *, epochs: int, seed: int) -> float:
    import torch
    from torch import nn

    n = int(y.shape[0])
    ys, mu, sd = _std_y(y)

    class LUT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e = nn.Embedding(n, 16)
            self.f = nn.Linear(16, 1)

        def forward(self, idx):
            return self.f(self.e(idx))

    idx = torch.arange(n, dtype=torch.long)
    hat_z = _fit_torch(LUT(), idx, ys, epochs=epochs, seed=seed, lr=5.0e-3)
    return _nrmse_denorm(hat_z, y, mu, sd)


def _mlp_nrmse(rgb64: np.ndarray, y: np.ndarray, *, epochs: int, seed: int) -> float:
    from torch import nn

    n = int(rgb64.shape[0])
    x = (np.asarray(rgb64, dtype=np.float32) / 255.0).reshape(n, -1)
    ys, mu, sd = _std_y(y)
    d = int(x.shape[1])
    model = nn.Sequential(nn.Linear(d, 1))
    hat_z = _fit_torch(model, x, ys, epochs=epochs, seed=seed, lr=1.0e-2)
    return _nrmse_denorm(hat_z, y, mu, sd)


def _gap_cnn():
    from torch import nn

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    return M()


def _coord_cnn():
    from torch import nn

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(5, 16, 5, stride=2, padding=2)
            self.c2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            import torch

            b, _, h, w = x.shape
            yy = torch_linspace(h, x.device).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch_linspace(w, x.device).view(1, 1, 1, w).expand(b, 1, h, w)
            z = torch.cat([x, xx, yy], dim=1)
            return self.fc(torch.relu(self.c2(torch.relu(self.c1(z)))))

    return M()


def torch_linspace(n: int, device: Any):
    import torch

    return torch.linspace(0.0, 1.0, n, device=device)


def _cnn_nrmse(rgb64: np.ndarray, y: np.ndarray, factory: Callable, *, epochs: int, seed: int) -> float:
    x = np.transpose(np.asarray(rgb64, dtype=np.float32) / 255.0, (0, 3, 1, 2))
    ys, mu, sd = _std_y(y)
    hat_z = _fit_torch(factory(), x, ys, epochs=epochs, seed=seed)
    return _nrmse_denorm(hat_z, y, mu, sd)


def _resize_n(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.stack([_resize_rgb(im, size) for im in rgb])


def _cam_list(env: Any) -> list[tuple[str, Any]]:
    c = env.cameras
    out: list[tuple[str, Any]] = []
    for cam, name in zip(c.static_camera_list, c.static_camera_name):
        out.append((str(name), cam))
    if getattr(c, "collect_wrist_camera", False):
        for name in ("left_camera", "right_camera"):
            if hasattr(c, name):
                out.append((name, getattr(c, name)))
    for name in ("observer_camera", "world_camera1", "world_camera2"):
        if hasattr(c, name):
            out.append((name, getattr(c, name)))
    # unique by name
    seen, uniq = set(), []
    for n, cam in out:
        if n in seen:
            continue
        seen.add(n)
        uniq.append((n, cam))
    return uniq


def _cam_wh(cam: Any) -> tuple[int, int]:
    w = int(getattr(cam, "width", 0) or getattr(cam, "w", 0) or 0)
    h = int(getattr(cam, "height", 0) or getattr(cam, "h", 0) or 0)
    if w <= 0 or h <= 0:
        try:
            K = np.asarray(cam.get_intrinsic_matrix())
            w, h = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
        except Exception:
            w, h = 640, 480
    return w, h


def collect_fov(cfg: RTWXO0D2Config) -> dict[str, Any]:
    if cfg.backend == "numpy":
        names = ["front_camera", "head_camera", "left_camera", "observer_camera"]
        return {
            "P": {n: 1.0 for n in names},
            "P_any": 1.0,
            "c_star": "front_camera",
            "n": int(cfg.n_train_ep) * int(cfg.n_steps),
        }
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_raster_shader()
    _patch_curobo_planner(repo)
    args = _o0_args(repo)
    rng = np.random.default_rng(cfg.seed)
    hits: dict[str, list[bool]] = {}
    n_ok = 0
    attempts, slot = 0, 0
    while n_ok < int(cfg.n_train_ep) and attempts < int(cfg.n_train_ep) * max(2, cfg.max_resample):
        attempts += 1
        seed = int(cfg.seed + slot * 17 + attempts)
        slot += 1
        print(f"[rtwx-o0d2] fov {n_ok}/{cfg.n_train_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            dt = 1.0 / 250.0
            tt = np.arange(cfg.n_steps) * (dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            specs = _cam_list(env)
            for name, _cam in specs:
                hits.setdefault(name, [])
            bad = False
            for i in range(int(cfg.n_steps)):
                p, _q = _cup_pose(env)
                any_hit = False
                for name, cam in specs:
                    w, h = _cam_wh(cam)
                    K = np.asarray(cam.get_intrinsic_matrix())
                    E = np.asarray(cam.get_extrinsic_matrix())
                    ok = bool(_project(K, E, p, w, h)["in_fov"])
                    hits[name].append(ok)
                    any_hit = any_hit or ok
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig
                if int(getattr(env, "take_action_cnt", 0)) == cnt0 or _contact_invalid(env):
                    bad = True
                    break
            env.close()
            if bad:
                for name in list(hits):
                    extra = len(hits[name]) % int(cfg.n_steps)
                    if extra:
                        hits[name] = hits[name][: len(hits[name]) - extra]
                continue
            n_ok += 1
        except Exception as exc:
            print(f"[rtwx-o0d2] fov fail {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if n_ok < int(cfg.n_train_ep):
        raise RuntimeError(f"O0D2 fov collect {n_ok}/{cfg.n_train_ep}")
    pmap = {n: float(np.mean(v)) if v else 0.0 for n, v in hits.items()}
    # any-camera: reconstruct per-frame OR from lists of equal length
    n_frame = min(len(v) for v in hits.values()) if hits else 0
    any_flags = []
    for t in range(n_frame):
        any_flags.append(any(hits[n][t] for n in hits))
    c_star = max(pmap, key=pmap.get) if pmap else ""
    return {"P": pmap, "P_any": float(np.mean(any_flags)) if any_flags else 0.0, "c_star": c_star, "n": n_frame}


def _pattern(*, a: bool, b: bool, gap_ok: bool, spatial_ok: bool, fov: bool) -> str:
    if not a:
        return "training_pipeline_failure"
    if not b:
        return "image_memorization_failure"
    if fov:
        return "instrument_closed"
    if spatial_ok and not gap_ok:
        return "spatial_representation_failure"
    return "coverage_insufficient"


def run_rtwx_o0d2(output: str | Path, config: RTWXO0D2Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0D2Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0D2",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "does_not_unlock_o1": True,
        "does_not_run_o0r": True,
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    o0 = None if cfg.smoke else _load_o0_train(Path(cfg.o0_cache))
    rng = np.random.default_rng(cfg.seed)
    if o0 is None:
        n = int(cfg.n_mem)
        ps, rgbs = [], []
        for i in range(n):
            p = np.array([-0.28 + 0.56 * i / max(n - 1, 1), rng.uniform(-0.2, 0.05), 0.71])
            rgb, _m = _blob_frame(p, 64)
            rgb = rgb.copy()
            rgb[0, i % 64, 2] = 40 + i
            ps.append(p)
            rgbs.append(rgb)
        px = np.stack(ps)[:, 0]
        rgb64 = np.stack(rgbs)
    else:
        n = min(int(cfg.n_mem), o0["p"].shape[0])
        idx = np.sort(rng.choice(o0["p"].shape[0], size=n, replace=False))
        px = o0["p"][idx, 0]
        rgb64 = _resize_n(o0["rgb"][idx], 64)
        # drop RGB collisions (same pixels cannot carry two random labels)
        keys, keep = set(), []
        for i, im in enumerate(rgb64):
            k = hash(im.tobytes())
            if k in keys:
                continue
            keys.add(k)
            keep.append(i)
        keep = np.asarray(keep)
        if keep.size < 8:
            raise RuntimeError("O0D2: too few unique RGB frames in subsample")
        if keep.size < n:
            rgb64, px = rgb64[keep], px[keep]
            n = int(keep.size)

    y_rand = (px.mean() + px.std() * rng.normal(size=n)).astype(np.float64)
    print("[rtwx-o0d2] D2-A lookup", flush=True)
    a_true = _lookup_nrmse(px, epochs=int(cfg.epoch_lut), seed=cfg.seed)
    a_rand = _lookup_nrmse(y_rand, epochs=int(cfg.epoch_lut), seed=cfg.seed + 1)
    a_ok = bool(a_true < LUT_MAX and a_rand < LUT_MAX)
    print(f"[rtwx-o0d2] lookup true={a_true:.4e} rand={a_rand:.4e} ok={a_ok}", flush=True)

    print("[rtwx-o0d2] D2-B flatten MLP", flush=True)
    b_true = _mlp_nrmse(rgb64, px, epochs=int(cfg.epoch_mlp), seed=cfg.seed)
    b_rand = _mlp_nrmse(rgb64, y_rand, epochs=int(cfg.epoch_mlp), seed=cfg.seed + 2)
    b_ok = bool(b_true < PIX_MAX and b_rand < PIX_MAX)
    print(f"[rtwx-o0d2] mlp true={b_true:.4f} rand={b_rand:.4f} ok={b_ok}", flush=True)

    print("[rtwx-o0d2] D2-C GAP vs CoordConv", flush=True)
    gap = _cnn_nrmse(rgb64, px, _gap_cnn, epochs=int(cfg.epoch_cnn), seed=cfg.seed)
    spat = _cnn_nrmse(rgb64, px, _coord_cnn, epochs=int(cfg.epoch_cnn), seed=cfg.seed + 3)
    gap_ok, spat_ok = bool(gap < PIX_MAX), bool(spat < PIX_MAX)
    print(f"[rtwx-o0d2] gap={gap:.4f} coord={spat:.4f}", flush=True)

    fov = collect_fov(cfg)
    pstar = float(fov["P"].get(fov["c_star"], 0.0)) if fov["P"] else 0.0
    fov_ok = bool(pstar >= FOV_MIN)
    print(f"[rtwx-o0d2] FOV c*={fov['c_star']} P={pstar:.3f} any={fov['P_any']:.3f}", flush=True)

    pattern = _pattern(a=a_ok, b=b_ok, gap_ok=gap_ok, spatial_ok=spat_ok, fov=fov_ok)
    summary = {
        "header": header,
        "pattern": pattern,
        "A_lookup": {"nrmse_true": a_true, "nrmse_rand": a_rand, "ok": a_ok},
        "B_flatten_mlp": {"nrmse_true": b_true, "nrmse_rand": b_rand, "ok": b_ok},
        "C_spatial": {"gap": gap, "coordconv": spat, "gap_ok": gap_ok, "spatial_ok": spat_ok, "spatial_bottleneck_on_GAP": bool(spat_ok and not gap_ok)},
        "FOV": {**fov, "P_star": pstar, "ok": fov_ok, "meets_0.9": [n for n, p in fov["P"].items() if p >= FOV_MIN]},
        "does_not_unlock_o1": True,
        "does_not_run_o0r": True,
        "unlocks_o0r": pattern == "instrument_closed",
        "no_observability_claim": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {k: summary[k] for k in ("pattern", "A_lookup", "B_flatten_mlp", "C_spatial", "FOV")},
    )
    return summary
