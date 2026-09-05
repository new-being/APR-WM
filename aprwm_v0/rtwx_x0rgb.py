"""RTWX-X0RGB: sim RGB → state → frozen M2 under K=4 deployment contract."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0c import (
    FORMAL_N_STEPS,
    FORMAL_N_TEST,
    FORMAL_N_TRAIN,
    FORMAL_N_VAL,
    _write_json,
)
from .rtwx_x0e import _apply_dx, _phi_m2, _x
from .rtwx_x0e1 import (
    BATCH,
    LR,
    PATIENCE,
    WD,
    _delta,
    _lstsq,
    _norm_stats,
)
from .rtwx_x0eh1 import (
    H_PLAN,
    ID_RATIO,
    K_EXECUTE,
    P_STRUCT,
    _bench_deploy,
    _m2_step_fn,
    eval_receding_k4,
)
from .rtwx_x0_smoke import TASK_NAME, _load_task_args, _patch_curobo_planner

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/rgb_robot_probe/RTWX0RGB_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0rgb.sim_rgb_state_m2.v1"
SEED_FORMAL = 14601
TASK = TASK_NAME
CAMERA = "front_camera"
L_HIST = 2
RGB_SIZE = 64
E_Q_MAX = 0.35
E_QD_MAX = 0.50
RETAIN_RATIO = 1.30
VIS_HIDDEN = 256
B2_HIDDEN = 256
VIS_EPOCHS = 40


@dataclass(frozen=True)
class RTWX0RGBConfig:
    output: str = "runs/rtwx_x0rgb"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED_FORMAL
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    dt_numpy: float = 0.05
    n_dof_numpy: int = 4
    l_hist: int = L_HIST
    rgb_size: int = RGB_SIZE
    vis_epochs: int = VIS_EPOCHS
    b2_hidden: int = B2_HIDDEN


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0RGB must not write there")


def _lock(cfg: RTWX0RGBConfig) -> RTWX0RGBConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=SEED_FORMAL,
        l_hist=L_HIST,
        rgb_size=RGB_SIZE,
        vis_epochs=VIS_EPOCHS,
    )


def _patch_raster_shader() -> None:
    """RoboTwin setup_scene forces RT+OIDN; OIDN is broken here and 32-spp RT is unusable.

    Raster RGB is still sim visual observation (not a real camera). Call before env setup.
    """
    import sapien.render as sr

    orig = sr.set_camera_shader_dir

    def _set(d: str, *a: Any, **k: Any) -> Any:
        if str(d) == "rt":
            d = "default"
        return orig(d, *a, **k)

    sr.set_camera_shader_dir = _set  # type: ignore[method-assign]
    for fn_name, val in (
        ("set_ray_tracing_denoiser", "none"),
        ("set_ray_tracing_samples_per_pixel", 1),
    ):
        fn = getattr(sr, fn_name, None)
        if callable(fn):
            try:
                fn(val)
            except Exception:
                pass


def _rgb_args(repo: Path) -> dict[str, Any]:
    args = _load_task_args(repo, "demo_clean")
    args["task_name"] = TASK
    args["render_freq"] = 0
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["qpos"] = False
    args["data_type"]["endpose"] = False
    args.setdefault("camera", {})
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = False
    return args


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 rgb, got {arr.shape}")
    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    return arr[ys][:, xs]


def _feat_from_q(q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    qn = qd / (np.max(np.abs(qd)) + 1.0e-6)
    return np.concatenate([np.sin(q), np.cos(q), qn], axis=-1).astype(np.float64)


def _synth_rgb_frame(q: np.ndarray, qd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    feat = _feat_from_q(q, qd)
    feat = feat + rng.normal(0.0, 0.05, size=feat.shape)
    side = int(np.ceil(np.sqrt(feat.size / 3.0)))
    pad = np.zeros(side * side * 3, dtype=np.float64)
    pad[: feat.size] = feat
    img = ((pad.reshape(side, side, 3) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return _resize_rgb(img, RGB_SIZE)


def _attach_synth_rgb(pool: dict[str, Any], rng: np.random.Generator) -> None:
    n = int(pool["q"].shape[0])
    rgb = np.zeros((n, RGB_SIZE, RGB_SIZE, 3), dtype=np.uint8)
    for i in range(n):
        rgb[i] = _synth_rgb_frame(pool["q"][i], pool["qd"][i], rng)
    pool["rgb"] = rgb
    pool["rgb_backend"] = "numpy_synth"


def _save_cache_rgb(path: Path, splits: dict[str, dict[str, Any]]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for name, p in splits.items():
        for k in ("q", "qd", "qn", "qdn", "q_tar"):
            payload[f"{name}_{k}"] = p[k]
        payload[f"{name}_rgb"] = p["rgb"]
        payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
        payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
        payload[f"{name}_n_dof"] = np.array([p["n_dof"]])
    np.savez_compressed(path, **payload)


def _save_one_split(path: Path, name: str, p: dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for k in ("q", "qd", "qn", "qdn", "q_tar"):
        payload[f"{name}_{k}"] = p[k]
    payload[f"{name}_rgb"] = p["rgb"]
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    payload[f"{name}_n_dof"] = np.array([p["n_dof"]])
    np.savez_compressed(path, **payload)


def _load_one_split(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    key = f"{name}_rgb"
    if key not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"], dtype=np.float64) for k in ("q", "qd", "qn", "qdn", "q_tar")}
    p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    p["n_dof"] = int(z[f"{name}_n_dof"][0])
    p["eq"] = p["q_tar"] - p["q"]
    p["source"] = "cache"
    return p


def _load_cache_rgb(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    z = np.load(path)
    if "train_rgb" not in z:
        return None
    out: dict[str, dict[str, Any]] = {}
    for name in ("train", "val", "test"):
        p = {k: np.asarray(z[f"{name}_{k}"], dtype=np.float64) for k in ("q", "qd", "qn", "qdn", "q_tar")}
        p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
        p["n_ep"] = int(z[f"{name}_n_ep"][0])
        p["n_steps"] = int(z[f"{name}_n_steps"][0])
        p["n_dof"] = int(z[f"{name}_n_dof"][0])
        p["eq"] = p["q_tar"] - p["q"]
        p["source"] = "cache"
        out[name] = p
    return out


def _stack_hist(flat_idx: int, ep_start: int, l_hist: int, frames: np.ndarray) -> np.ndarray:
    t = flat_idx - ep_start
    out: list[np.ndarray] = []
    for k in range(l_hist):
        tk = max(0, t - (l_hist - 1 - k))
        out.append(frames[ep_start + tk].reshape(-1))
    return np.concatenate(out, axis=0).astype(np.float64)


def _build_vis_mlp(in_dim: int, out_dim: int, hidden: int):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _train_vis(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    *,
    out_dim: int,
    hidden: int,
    seed: int,
    epochs: int,
) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_vis_mlp(int(xtr.shape[1]), out_dim, hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    tr_ds = TensorDataset(torch.from_numpy(xtr).float(), torch.from_numpy(ytr).float())
    va_ds = TensorDataset(torch.from_numpy(xva).float(), torch.from_numpy(yva).float())
    tr_ld = DataLoader(tr_ds, batch_size=min(BATCH, len(tr_ds)), shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=min(BATCH, len(va_ds)), shuffle=False)
    best = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for xb, yb in va_ld:
                vals.append(float(loss_fn(model(xb.to(dev)), yb.to(dev)).item()))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1.0e-6:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _vis_np(model: Any, x: np.ndarray) -> np.ndarray:
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.from_numpy(x).float().to(dev)).cpu().numpy()


def _make_vis_dataset(pool: dict[str, Any], l_hist: int) -> tuple[np.ndarray, np.ndarray]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    rgb = pool["rgb"]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in range(n_ep):
        s0 = ep * n_steps
        for t in range(n_steps):
            idx = s0 + t
            xs.append(_stack_hist(idx, s0, l_hist, rgb))
            ys.append(np.concatenate([pool["q"][idx], pool["qd"][idx]]))
    return np.stack(xs), np.stack(ys)


def _perception_metrics(pool: dict[str, Any], model: Any, l_hist: int) -> dict[str, float]:
    x, y = _make_vis_dataset(pool, l_hist)
    pred = _vis_np(model, x)
    d = y.shape[1] // 2
    return {
        "E_q": _nrmse(pred[:, :d], y[:, :d]),
        "E_qd": _nrmse(pred[:, d:], y[:, d:]),
    }


def _vis_reset_fn(pool: dict[str, Any], model: Any, l_hist: int):
    n_steps = int(pool["n_steps"])

    def reset(ep: int, t0: int) -> tuple[np.ndarray, np.ndarray]:
        s0 = ep * n_steps
        idx = s0 + t0
        x = _stack_hist(idx, s0, l_hist, pool["rgb"]).reshape(1, -1)
        pred = _vis_np(model, x).reshape(-1)
        d = pred.size // 2
        return pred[:d].copy(), pred[d:].copy()

    return reset


def eval_receding_k4_vis(
    pool: dict[str, Any],
    step_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    reset_fn: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    scale: float,
    *,
    k: int = K_EXECUTE,
    oracle_reset: bool = False,
) -> dict[str, Any]:
    from .rtwx_x0es import CATA_EH

    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    segs: list[dict[str, Any]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        t0 = 0
        while t0 + k <= n_steps:
            if oracle_reset:
                qh, qdh = qq[t0].copy(), qddt[t0].copy()
            else:
                qh, qdh = reset_fn(ep, t0)

            preds: list[np.ndarray] = []
            refs: list[np.ndarray] = []
            mx = 0.0
            finite = True

            for h in range(k):
                qh, qdh = step_fn(qh, qdh, qt[t0 + h])
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    finite = False
                    mx = float("inf")
                    break
                pred = _x(qh, qdh)
                ref = _x(qnn[t0 + h], qdnn[t0 + h])
                preds.append(pred)
                refs.append(ref)
                mx = max(mx, float(np.linalg.norm(pred - ref) / (scale + 1.0e-8)))
            cata = (not finite) or mx > CATA_EH
            if preds:
                e_exec = _nrmse(np.stack(preds), np.stack(refs))
                e_end = _nrmse(preds[-1], refs[-1]) if finite else float("inf")
            else:
                e_exec = float("inf")
                e_end = float("inf")
            x0 = _x(qq[t0], qddt[t0]) if oracle_reset else _x(*reset_fn(ep, t0))
            e_id = _nrmse(
                np.stack([x0 for _ in range(k)]),
                np.stack([_x(qnn[t0 + h], qdnn[t0 + h]) for h in range(k)]),
            )
            segs.append({"cata": cata, "nonfinite": not finite, "e_exec": e_exec, "e_end": e_end, "e_id": e_id})
            t0 += k
    n = len(segs)
    n_cata = sum(1 for s in segs if s["cata"])
    e_execs = np.array([s["e_exec"] for s in segs], dtype=np.float64)
    e_ends = np.array([s["e_end"] for s in segs], dtype=np.float64)
    e_ids = np.array([s["e_id"] for s in segs], dtype=np.float64)

    def _mean(a: np.ndarray) -> float:
        if a.size == 0 or not np.isfinite(a).all():
            return float("inf")
        return float(np.mean(a))

    return {
        "K": k,
        "n_segments": n,
        "n_cata": n_cata,
        "n_nonfinite": sum(1 for s in segs if s["nonfinite"]),
        "r_cat": float(n_cata / max(n, 1)),
        "E_exec": _mean(e_execs),
        "E_end": _mean(e_ends),
        "E_identity": _mean(e_ids),
    }


def _train_b2(
    pool: dict[str, Any],
    val: dict[str, Any],
    l_hist: int,
    hidden: int,
    seed: int,
    epochs: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    xtr, ytr = _make_b2_xy(pool, l_hist)
    xva, yva = _make_b2_xy(val, l_hist)
    mu, sd = _norm_stats(xtr)
    sd = np.where(sd < 1.0e-8, 1.0, sd)
    xtr_n = (xtr - mu) / sd
    xva_n = (xva - mu) / sd
    model = _train_vis(xtr_n, ytr, xva_n, yva, out_dim=ytr.shape[1], hidden=hidden, seed=seed, epochs=epochs)
    return model, mu, sd


def _make_b2_xy(pool: dict[str, Any], l_hist: int) -> tuple[np.ndarray, np.ndarray]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for ep in range(n_ep):
        s0 = ep * n_steps
        for t in range(n_steps - 1):
            idx = s0 + t
            xin = np.concatenate([_stack_hist(idx, s0, l_hist, pool["rgb"]), pool["q_tar"][idx]])
            xs.append(xin)
            ys.append(np.concatenate([pool["qn"][idx] - pool["q"][idx], pool["qdn"][idx] - pool["qd"][idx]]))
    return np.stack(xs), np.stack(ys)


def eval_receding_k4_b2(
    pool: dict[str, Any],
    vis_model: Any,
    dyn_model: Any,
    mu: np.ndarray,
    sd: np.ndarray,
    l_hist: int,
    scale: float,
    *,
    k: int = K_EXECUTE,
) -> dict[str, Any]:
    n_steps = int(pool["n_steps"])
    n_dof = int(pool["n_dof"])
    reset_vis = _vis_reset_fn(pool, vis_model, l_hist)
    from .rtwx_x0es import CATA_EH

    n_ep = int(pool["n_ep"])
    q, qd, qn, qdn, qtar = pool["q"], pool["qd"], pool["qn"], pool["qdn"], pool["q_tar"]
    segs: list[dict[str, Any]] = []
    for ep in range(n_ep):
        sl = slice(ep * n_steps, (ep + 1) * n_steps)
        qq, qddt, qnn, qdnn, qt = q[sl], qd[sl], qn[sl], qdn[sl], qtar[sl]
        s0 = ep * n_steps
        t0 = 0
        while t0 + k <= n_steps:
            qh, qdh = reset_vis(ep, t0)
            preds, refs = [], []
            mx = 0.0
            finite = True
            for h in range(k):
                idx = s0 + t0 + h
                xin = np.concatenate([_stack_hist(idx, s0, l_hist, pool["rgb"]), qt[t0 + h]])
                z = ((xin - mu) / sd).reshape(1, -1)
                d = _vis_np(dyn_model, z).reshape(-1)
                qh, qdh = _apply_dx(qh, qdh, d)
                if not np.isfinite(qh).all() or not np.isfinite(qdh).all():
                    finite = False
                    mx = float("inf")
                    break
                pred = _x(qh, qdh)
                ref = _x(qnn[t0 + h], qdnn[t0 + h])
                preds.append(pred)
                refs.append(ref)
                mx = max(mx, float(np.linalg.norm(pred - ref) / (scale + 1.0e-8)))
            cata = (not finite) or mx > CATA_EH
            if preds:
                e_exec = _nrmse(np.stack(preds), np.stack(refs))
                e_end = _nrmse(preds[-1], refs[-1]) if finite else float("inf")
            else:
                e_exec = float("inf")
                e_end = float("inf")
            rq, rqd = reset_vis(ep, t0)
            e_id = _nrmse(
                np.stack([_x(rq, rqd) for _ in range(k)]),
                np.stack([_x(qnn[t0 + h], qdnn[t0 + h]) for h in range(k)]),
            )
            segs.append({"cata": cata, "nonfinite": not finite, "e_exec": e_exec, "e_end": e_end, "e_id": e_id})
            t0 += k
    n = len(segs)
    n_cata = sum(1 for s in segs if s["cata"])

    def _mean(key: str) -> float:
        a = np.array([s[key] for s in segs], dtype=np.float64)
        if a.size == 0 or not np.isfinite(a).all():
            return float("inf")
        return float(np.mean(a))

    return {
        "K": k,
        "n_segments": n,
        "n_cata": n_cata,
        "n_nonfinite": sum(1 for s in segs if s["nonfinite"]),
        "r_cat": float(n_cata / max(n, 1)),
        "E_exec": _mean("e_exec"),
        "E_end": _mean("e_end"),
        "E_identity": _mean("e_id"),
    }


def _competent(m: dict[str, Any]) -> bool:
    return bool(
        m["r_cat"] == 0.0
        and m["n_nonfinite"] == 0
        and np.isfinite(m["E_exec"])
        and np.isfinite(m["E_identity"])
        and m["E_identity"] > 0
        and m["E_exec"] <= ID_RATIO * m["E_identity"]
    )


def _pattern(g_oracle: bool, g_vis: bool, g_deploy: bool, g_retain: bool, b1: dict[str, Any]) -> str:
    if not g_oracle:
        return "oracle_regression"
    if not g_vis:
        return "perception_failure"
    if float(b1["r_cat"]) > 0.0:
        return "catastrophe_returns"
    if g_deploy and g_retain:
        return "structure_survives_sim_rgb"
    if g_deploy:
        return "vision_limited_structure"
    return "vision_limited_structure"


def _collect_numpy(cfg: RTWX0RGBConfig, split: str, rng: np.random.Generator) -> dict[str, Any]:
    from .rtwx_x0e1 import RTWX0E1Config, collect_numpy_x0e1

    c = RTWX0E1Config(
        backend="numpy",
        n_train_ep=cfg.n_train_ep,
        n_val_ep=cfg.n_val_ep,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        dt_numpy=cfg.dt_numpy,
        n_dof_numpy=cfg.n_dof_numpy,
        smoke=True,
    )
    pool = collect_numpy_x0e1(c, split=split, rng=rng)
    _attach_synth_rgb(pool, rng)
    pool["eq"] = pool["q_tar"] - pool["q"]
    return pool


def _capture_rgb_step(env: Any, size: int) -> np.ndarray:
    env._update_render()
    env.cameras.update_picture()
    rgb_map = env.cameras.get_rgb()
    cam = rgb_map.get(CAMERA) or rgb_map.get("head_camera")
    if cam is None or cam.get("rgb") is None:
        raise RuntimeError(f"missing {CAMERA}/head_camera rgb; keys={list(rgb_map)}")
    return _resize_rgb(cam["rgb"], size)


def collect_robotwin_rgb(
    cfg: RTWX0RGBConfig,
    *,
    split: str,
    rng: np.random.Generator,
    stop_mode: list[str],
) -> dict[str, Any]:
    """Oracle joint trajectories + front_camera RGB per step."""
    from .rtwx_x0c import (
        _arm_q,
        _arm_qd,
        _build_action,
        _contact_invalid,
        _count_steps,
        _drive_pack,
        _joint_limits,
        _multisine,
    )
    from .rtwx_x0 import _setup_env as setup

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _rgb_args(repo)
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = cfg.seed + {"train": 0, "val": 10_000, "test": 20_000}[split]
    eps: list[dict[str, Any]] = []
    reasons: list[str] = []
    attempts_used = 0
    slot = 0
    while len(eps) < n_ep and attempts_used < n_ep * max(2, cfg.max_resample):
        attempts_used += 1
        seed = int(seed0 + slot * 17 + attempts_used)
        slot += 1
        if len(eps) % 2 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-x0rgb] {split} valid {len(eps)}/{n_ep} attempt {attempts_used}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            if getattr(env, "step_lim", 1000) is None or int(env.step_lim) <= int(cfg.n_steps):
                env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            if _contact_invalid(env):
                reasons.append("contact_at_start")
                env.close()
                continue
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            for getter in ("get_timestep",):
                fn = getattr(env.scene, getter, None)
                if callable(fn):
                    try:
                        scene_dt = float(fn())
                        break
                    except Exception:
                        pass
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            dt_guess = scene_dt * 50.0
            t = np.arange(cfg.n_steps, dtype=np.float64) * dt_guess
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, t, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, t, rng)
            rows: dict[str, list] = {k: [] for k in ("q", "qd", "qn", "qdn", "q_tar", "rgb")}
            invalid = None
            for i in range(int(cfg.n_steps)):
                q_l, q_r = _arm_q(env, "left"), _arm_q(env, "right")
                qd_l, qd_r = _arm_qd(env, "left"), _arm_qd(env, "right")
                drv_pre = _drive_pack(jl + jr)
                rows["rgb"].append(_capture_rgb_step(env, cfg.rgb_size))
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig_step, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig_step
                cnt1 = int(getattr(env, "take_action_cnt", 0))
                if cnt1 == cnt0:
                    invalid = "take_action_noop"
                    break
                drv_post = _drive_pack(jl + jr)
                q_l2, q_r2 = _arm_q(env, "left"), _arm_q(env, "right")
                qd_l2, qd_r2 = _arm_qd(env, "left"), _arm_qd(env, "right")
                n_l = q_l.size
                cmd_l = np.asarray(action[:n_l], dtype=np.float64)
                cmd_r = np.asarray(action[n_l + 1 : n_l + 1 + q_r.size], dtype=np.float64)
                moved_l = not np.allclose(drv_post["q_tar"][:n_l], drv_pre["q_tar"][:n_l], atol=1e-5)
                moved_r = not np.allclose(drv_post["q_tar"][n_l:], drv_pre["q_tar"][n_l:], atol=1e-5)
                match_l = np.allclose(drv_post["q_tar"][:n_l], cmd_l, atol=5e-3)
                match_r = np.allclose(drv_post["q_tar"][n_l:], cmd_r, atol=5e-3)
                if (not moved_l and not match_l) or (not moved_r and not match_r):
                    invalid = "topp_skip_drive_not_updated"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
                q = np.concatenate([q_l, q_r])
                qd = np.concatenate([qd_l, qd_r])
                q_tar = np.concatenate([cmd_l, cmd_r])
                rows["q"].append(q)
                rows["qd"].append(qd)
                rows["qn"].append(np.concatenate([q_l2, q_r2]))
                rows["qdn"].append(np.concatenate([qd_l2, qd_r2]))
                rows["q_tar"].append(q_tar)
            try:
                env.close()
            except Exception:
                pass
            if invalid is not None:
                reasons.append(invalid)
                continue
            ep = {
                "q": np.asarray(rows["q"], dtype=np.float64),
                "qd": np.asarray(rows["qd"], dtype=np.float64),
                "qn": np.asarray(rows["qn"], dtype=np.float64),
                "qdn": np.asarray(rows["qdn"], dtype=np.float64),
                "q_tar": np.asarray(rows["q_tar"], dtype=np.float64),
                "rgb": np.stack(rows["rgb"], axis=0),
            }
            if ep["q"].shape[0] != int(cfg.n_steps):
                reasons.append("short_episode")
                continue
            eps.append(ep)
        except Exception as exc:
            reasons.append(f"setup:{type(exc).__name__}:{exc}")
            print(f"[rtwx-x0rgb] {split} fail attempt {attempts_used}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            continue
        if attempts_used % 8 == 0 and reasons:
            print(f"[rtwx-x0rgb] {split} recent reasons={reasons[-5:]}", flush=True)
    if len(eps) < n_ep:
        stop_mode.append(f"{split}:only_{len(eps)}_of_{n_ep}")
        print(f"[rtwx-x0rgb] {split} incomplete reasons={reasons[-20:]}", flush=True)
    if not eps:
        raise RuntimeError(f"X0RGB collect {split}: 0 valid episodes; reasons={reasons[-20:]}")
    keys = ("q", "qd", "qn", "qdn", "q_tar", "rgb")
    out: dict[str, Any] = {
        k: np.concatenate([e[k] for e in eps], axis=0)
        for k in keys
    }
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    out["n_dof"] = int(eps[0]["q"].shape[1]) if eps else 0
    out["rgb_backend"] = "robotwin_front_camera"
    out["eq"] = out["q_tar"] - out["q"]
    return out


def _collect(cfg: RTWX0RGBConfig, root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    cache = root / "cache_rgb_splits.npz"
    hit = _load_cache_rgb(cache)
    if hit is not None:
        return hit
    stop: list[str] = []
    rng_tr = np.random.default_rng(cfg.seed)
    rng_va = np.random.default_rng(cfg.seed + 1)
    rng_te = np.random.default_rng(cfg.seed + 2)
    splits: dict[str, dict[str, Any]] = {}
    for name, rng in (("train", rng_tr), ("val", rng_va), ("test", rng_te)):
        part = root / f"cache_rgb_{name}.npz"
        loaded = _load_one_split(part, name)
        if loaded is not None:
            print(f"[rtwx-x0rgb] {name}: using split cache", flush=True)
            splits[name] = loaded
            continue
        if cfg.backend == "numpy":
            pool = _collect_numpy(cfg, name, rng)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            _patch_raster_shader()
            pool = collect_robotwin_rgb(cfg, split=name, rng=rng, stop_mode=stop)
        splits[name] = pool
        _save_one_split(part, name, pool)
    if stop:
        raise RuntimeError(f"X0RGB collect stop: {stop}")
    _save_cache_rgb(cache, splits)
    return splits


def run_rtwx_x0rgb(output: str | Path, config: RTWX0RGBConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0RGBConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-X0RGB",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "camera": CAMERA,
        "l_hist": cfg.l_hist,
        "rgb_size": cfg.rgb_size,
        "deployment_contract": {"H_plan": H_PLAN, "K_execute": K_EXECUTE},
        "seed": cfg.seed,
        "sim_rgb_not_real_camera": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    splits = _collect(cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    ytr = _delta(train)
    w_m2 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), ytr)
    if not cfg.smoke and int(w_m2.size) != P_STRUCT:
        raise RuntimeError(f"M2 param count {w_m2.size} != {P_STRUCT}")
    np.save(root / "W_m2.npy", w_m2)

    xtr, ytr_vis = _make_vis_dataset(train, cfg.l_hist)
    xva, yva_vis = _make_vis_dataset(val, cfg.l_hist)
    out_dim = int(ytr_vis.shape[1])
    vis_model = _train_vis(
        xtr, ytr_vis, xva, yva_vis,
        out_dim=out_dim,
        hidden=VIS_HIDDEN if not cfg.smoke else 64,
        seed=cfg.seed,
        epochs=cfg.vis_epochs if not cfg.smoke else 5,
    )
    perc_test = _perception_metrics(test, vis_model, cfg.l_hist)
    print(f"[rtwx-x0rgb] perception E_q={perc_test['E_q']:.4f} E_qd={perc_test['E_qd']:.4f}", flush=True)

    b0_test = eval_receding_k4(test, _m2_step_fn(w_m2), scale, k=K_EXECUTE)
    vis_reset = _vis_reset_fn(test, vis_model, cfg.l_hist)
    b1_test = eval_receding_k4_vis(
        test,
        _m2_step_fn(w_m2),
        vis_reset,
        scale,
        k=K_EXECUTE,
        oracle_reset=False,
    )
    print(f"[rtwx-x0rgb] B0 E_exec={b0_test['E_exec']:.4f} B1 E_exec={b1_test['E_exec']:.4f}", flush=True)

    b2_model, b2_mu, b2_sd = _train_b2(train, val, cfg.l_hist, cfg.b2_hidden if not cfg.smoke else 64, cfg.seed, cfg.vis_epochs if not cfg.smoke else 5)
    b2_test = eval_receding_k4_b2(test, vis_model, b2_model, b2_mu, b2_sd, cfg.l_hist, scale, k=K_EXECUTE)

    g_oracle = _competent(b0_test)
    g_vis = bool(perc_test["E_q"] <= E_Q_MAX and perc_test["E_qd"] <= E_QD_MAX) if not cfg.smoke else bool(np.isfinite(perc_test["E_q"]))
    g_deploy = bool(b1_test["r_cat"] == 0.0 and b1_test["n_nonfinite"] == 0)
    g_retain = bool(np.isfinite(b0_test["E_exec"]) and b1_test["E_exec"] <= RETAIN_RATIO * b0_test["E_exec"])
    pattern = _pattern(g_oracle, g_vis, g_deploy, g_retain, b1_test)

    delta_vis = float(b1_test["E_exec"] - b0_test["E_exec"]) if np.isfinite(b0_test["E_exec"]) else None
    denom = float(b0_test["E_identity"] - b0_test["E_exec"])
    eta_retain = (
        float((b0_test["E_identity"] - b1_test["E_exec"]) / denom)
        if np.isfinite(denom) and denom > 1.0e-8
        else None
    )

    ep_sl = slice(0, int(test["n_steps"]))
    deploy = {
        "B0_oracle_M2": _bench_deploy(_m2_step_fn(w_m2), test["q"][ep_sl], test["qd"][ep_sl], test["q_tar"][ep_sl], n_plan=H_PLAN, k=K_EXECUTE),
    }

    summary = {
        "header": header,
        "pattern": pattern,
        "G_oracle": g_oracle,
        "G_vis": g_vis,
        "G_deploy": g_deploy,
        "G_retain": g_retain,
        "perception_test": perc_test,
        "B0_oracle": {"test_K4": b0_test, "P": int(w_m2.size)},
        "B1_rgb_m2": {"test_K4": b1_test, "P_struct": int(w_m2.size), "vis_hidden": VIS_HIDDEN if not cfg.smoke else 64},
        "B2_rgb_latent_exploratory": {"test_K4": b2_test, "hidden": cfg.b2_hidden},
        "delta_vision": delta_vis,
        "eta_retain_exploratory": eta_retain,
        "deploy_compute": deploy,
        "structure_survives_sim_rgb_claim": pattern == "structure_survives_sim_rgb",
        "does_not_claim_real_camera": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "G_oracle", "G_vis", "G_deploy", "G_retain", "perception_test", "delta_vision")} | {"B0": b0_test, "B1": b1_test})
    return summary
