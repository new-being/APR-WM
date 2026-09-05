"""RTWX-X0RGB1: spatial-temporal RGB → (q, qd). Frozen M2. No capacity claim."""

from __future__ import annotations

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
from .rtwx_x0e import _phi_m2
from .rtwx_x0e1 import LR, PATIENCE, WD, _delta, _lstsq, _x
from .rtwx_x0eh1 import ID_RATIO, K_EXECUTE, P_STRUCT, _m2_step_fn, eval_receding_k4
from .rtwx_x0rgb import (
    E_QD_MAX,
    E_Q_MAX,
    CAMERA,
    TASK,
    VIS_HIDDEN,
    VIS_EPOCHS,
    RTWX0RGBConfig,
    _attach_synth_rgb,
    _collect,
    _perception_metrics,
    _resize_rgb,
    _train_vis,
    eval_receding_k4_vis,
)

PREREG_PATH = "REPORT/REG/RTWX/robot_dynamics/rgb_robot_probe/RTWX0RGB1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_x0rgb1.spatial_temporal_state.v1"
SEED_FORMAL = 15601
RGB_SIZE = 128
L_TEMP = 4
L_P0 = 2
P0_SIZE = 64
LAMBDA_V = 1.0
CNN_CH = (16, 32, 64, 128)


@dataclass(frozen=True)
class RTWX0RGB1Config:
    output: str = "runs/rtwx_x0rgb1"
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
    rgb_size: int = RGB_SIZE
    vis_epochs: int = VIS_EPOCHS
    cnn_ch: tuple[int, ...] = CNN_CH


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-X0RGB1 must not write there")


def _lock(cfg: RTWX0RGB1Config) -> RTWX0RGB1Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=SEED_FORMAL,
        rgb_size=RGB_SIZE,
        vis_epochs=VIS_EPOCHS,
        cnn_ch=CNN_CH,
    )


def _to_rgb_cfg(cfg: RTWX0RGB1Config) -> RTWX0RGBConfig:
    return RTWX0RGBConfig(
        output=cfg.output,
        robotwin_repo=cfg.robotwin_repo,
        backend=cfg.backend,
        n_train_ep=cfg.n_train_ep,
        n_val_ep=cfg.n_val_ep,
        n_test_ep=cfg.n_test_ep,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
        seed_attempts=cfg.seed_attempts,
        max_resample=cfg.max_resample,
        smoke=cfg.smoke,
        dt_numpy=cfg.dt_numpy,
        n_dof_numpy=cfg.n_dof_numpy,
        rgb_size=cfg.rgb_size,
        vis_epochs=cfg.vis_epochs,
    )


def _downsample_pool(pool: dict[str, Any], size: int) -> dict[str, Any]:
    out = dict(pool)
    rgb = np.asarray(pool["rgb"])
    out["rgb"] = np.stack([_resize_rgb(rgb[i], size) for i in range(rgb.shape[0])], axis=0)
    return out


def _state_stats(q: np.ndarray, qd: np.ndarray) -> dict[str, np.ndarray]:
    mu_q, sd_q = q.mean(axis=0), q.std(axis=0)
    mu_d, sd_d = qd.mean(axis=0), qd.std(axis=0)
    sd_q = np.where(sd_q < 1.0e-8, 1.0, sd_q)
    sd_d = np.where(sd_d < 1.0e-8, 1.0, sd_d)
    return {"mu_q": mu_q, "sd_q": sd_q, "mu_qd": mu_d, "sd_qd": sd_d}


def _norm(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _denorm(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return x * sd + mu


def _per_joint(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.asarray([_nrmse(pred[:, j], ref[:, j]) for j in range(ref.shape[1])], dtype=np.float64)


def _rgb_nchw(frames: np.ndarray) -> np.ndarray:
    x = np.asarray(frames, dtype=np.float32) / 255.0
    if x.ndim == 3:
        x = x[None, ...]
    return np.transpose(x, (0, 3, 1, 2))


def _clip_hist(rgb: np.ndarray, ep_start: int, t: int, l_hist: int) -> np.ndarray:
    idx = [ep_start + max(0, t - (l_hist - 1 - k)) for k in range(l_hist)]
    return rgb[np.asarray(idx)]


def _make_backbone(ch: tuple[int, ...]):
    from torch import nn

    layers: list[Any] = []
    cin = 3
    for c in ch:
        layers += [nn.Conv2d(cin, c, 3, stride=2, padding=1), nn.SiLU()]
        cin = c
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers), int(ch[-1])


def _build_p1(n_dof: int, ch: tuple[int, ...]):
    from torch import nn

    bb, feat = _make_backbone(ch)

    class P1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bb = bb
            self.hq = nn.Linear(feat, n_dof)
            self.hd = nn.Linear(feat, n_dof)

        def forward(self, x):
            f = self.bb(x).flatten(1)
            return self.hq(f), self.hd(f)

    return P1()


def _build_p2(n_dof: int, ch: tuple[int, ...], l_hist: int):
    from torch import nn

    bb, feat = _make_backbone(ch)

    class P2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bb = bb
            self.l_hist = l_hist
            self.hq = nn.Linear(feat, n_dof)
            self.tv = nn.Sequential(
                nn.Linear(feat * l_hist, feat),
                nn.SiLU(),
                nn.Linear(feat, n_dof),
            )

        def forward(self, x):
            b, l, c, h, w = x.shape
            f = self.bb(x.reshape(b * l, c, h, w)).flatten(1).reshape(b, l, -1)
            q = self.hq(f[:, -1])
            qd = self.tv(f.reshape(b, -1))
            return q, qd

    return P2()


def _train_split_heads(
    model: Any,
    batches_tr: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batches_va: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    seed: int,
    epochs: int,
    lambda_v: float = LAMBDA_V,
) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + _)
        order = rng.permutation(len(batches_tr))
        for i in order:
            xb, yq, yd = batches_tr[int(i)]
            xb_t = torch.from_numpy(xb).to(dev)
            yq_t = torch.from_numpy(yq).float().to(dev)
            yd_t = torch.from_numpy(yd).float().to(dev)
            opt.zero_grad()
            hq, hd = model(xb_t)
            loss = loss_fn(hq, yq_t) + lambda_v * loss_fn(hd, yd_t)
            loss.backward()
            opt.step()
        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for xb, yq, yd in batches_va:
                xb_t = torch.from_numpy(xb).to(dev)
                yq_t = torch.from_numpy(yq).float().to(dev)
                yd_t = torch.from_numpy(yd).float().to(dev)
                hq, hd = model(xb_t)
                vals.append(float((loss_fn(hq, yq_t) + lambda_v * loss_fn(hd, yd_t)).item()))
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


def _iter_p1(pool: dict[str, Any], stats: dict[str, np.ndarray], bs: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = int(pool["q"].shape[0])
    qn = _norm(pool["q"], stats["mu_q"], stats["sd_q"]).astype(np.float32)
    dn = _norm(pool["qd"], stats["mu_qd"], stats["sd_qd"]).astype(np.float32)
    out = []
    for i0 in range(0, n, bs):
        sl = slice(i0, min(n, i0 + bs))
        out.append((_rgb_nchw(pool["rgb"][sl]).astype(np.float32), qn[sl], dn[sl]))
    return out


def _iter_p2(pool: dict[str, Any], stats: dict[str, np.ndarray], l_hist: int, bs: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    qn = _norm(pool["q"], stats["mu_q"], stats["sd_q"]).astype(np.float32)
    dn = _norm(pool["qd"], stats["mu_qd"], stats["sd_qd"]).astype(np.float32)
    xs, yq, yd = [], [], []
    for ep in range(n_ep):
        s0 = ep * n_steps
        for t in range(n_steps):
            clip = _clip_hist(pool["rgb"], s0, t, l_hist)
            xs.append(_rgb_nchw(clip))
            yq.append(qn[s0 + t])
            yd.append(dn[s0 + t])
    x = np.stack(xs, axis=0).astype(np.float32)
    yq_a = np.stack(yq, axis=0)
    yd_a = np.stack(yd, axis=0)
    out = []
    for i0 in range(0, x.shape[0], bs):
        sl = slice(i0, min(x.shape[0], i0 + bs))
        out.append((x[sl], yq_a[sl], yd_a[sl]))
    return out


def _predict_p1(model: Any, pool: dict[str, Any], stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    qs, ds = [], []
    model.eval()
    with torch.no_grad():
        for i0 in range(0, pool["rgb"].shape[0], 64):
            sl = slice(i0, min(pool["rgb"].shape[0], i0 + 64))
            x = torch.from_numpy(_rgb_nchw(pool["rgb"][sl])).to(dev)
            hq, hd = model(x)
            qs.append(_denorm(hq.cpu().numpy(), stats["mu_q"], stats["sd_q"]))
            ds.append(_denorm(hd.cpu().numpy(), stats["mu_qd"], stats["sd_qd"]))
    return np.concatenate(qs, axis=0), np.concatenate(ds, axis=0)


def _predict_p2(model: Any, pool: dict[str, Any], stats: dict[str, np.ndarray], l_hist: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    qs, ds = [], []
    model.eval()
    batch_x: list[np.ndarray] = []
    with torch.no_grad():
        def flush() -> None:
            if not batch_x:
                return
            x = torch.from_numpy(np.stack(batch_x, axis=0)).to(dev)
            hq, hd = model(x)
            qs.append(_denorm(hq.cpu().numpy(), stats["mu_q"], stats["sd_q"]))
            ds.append(_denorm(hd.cpu().numpy(), stats["mu_qd"], stats["sd_qd"]))
            batch_x.clear()

        for ep in range(n_ep):
            s0 = ep * n_steps
            for t in range(n_steps):
                batch_x.append(_rgb_nchw(_clip_hist(pool["rgb"], s0, t, l_hist)))
                if len(batch_x) >= 32:
                    flush()
        flush()
    return np.concatenate(qs, axis=0), np.concatenate(ds, axis=0)


def _audit(pred_q: np.ndarray, pred_d: np.ndarray, pool: dict[str, Any]) -> dict[str, Any]:
    eqj = _per_joint(pred_q, pool["q"])
    edj = _per_joint(pred_d, pool["qd"])
    return {
        "E_q": _nrmse(pred_q, pool["q"]),
        "E_qd": _nrmse(pred_d, pool["qd"]),
        "E_q_j": eqj.tolist(),
        "E_qd_j": edj.tolist(),
        "n_q_below_035": int(np.sum(eqj < E_Q_MAX)),
        "n_dof": int(eqj.size),
    }


def _p2_reset_fn(pool: dict[str, Any], model: Any, stats: dict[str, np.ndarray], l_hist: int):
    n_steps = int(pool["n_steps"])

    def reset(ep: int, t0: int) -> tuple[np.ndarray, np.ndarray]:
        s0 = ep * n_steps
        return _predict_p2_one(model, pool, stats, l_hist, s0, t0)

    return reset


def _predict_p2_one(
    model: Any, pool: dict[str, Any], stats: dict[str, np.ndarray], l_hist: int, s0: int, t: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    x = torch.from_numpy(_rgb_nchw(_clip_hist(pool["rgb"], s0, t, l_hist))[None]).to(dev)
    model.eval()
    with torch.no_grad():
        hq, hd = model(x)
    q = _denorm(hq.cpu().numpy().reshape(-1), stats["mu_q"], stats["sd_q"])
    d = _denorm(hd.cpu().numpy().reshape(-1), stats["mu_qd"], stats["sd_qd"])
    return q, d


def _competent(m: dict[str, Any]) -> bool:
    return bool(
        m["r_cat"] == 0.0
        and m["n_nonfinite"] == 0
        and np.isfinite(m["E_exec"])
        and np.isfinite(m["E_identity"])
        and m["E_identity"] > 0
        and m["E_exec"] <= ID_RATIO * m["E_identity"]
    )


def _pattern_p2(perc: dict[str, Any], vis_deploy: dict[str, Any] | None) -> str:
    eq, ed = float(perc["E_q"]), float(perc["E_qd"])
    if eq > E_Q_MAX:
        return "spatial_perception_failure"
    if ed > E_QD_MAX:
        return "velocity_perception_failure"
    if vis_deploy is None:
        return "visual_dynamics_interface_failure"
    if float(vis_deploy["r_cat"]) > 0.0 or not _competent(vis_deploy):
        return "visual_dynamics_interface_failure"
    return "sim_rgb_state_interface_supported"


def run_rtwx_x0rgb1(output: str | Path, config: RTWX0RGB1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWX0RGB1Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-X0RGB1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "camera": CAMERA,
        "rgb_size": cfg.rgb_size,
        "L_temporal": L_TEMP,
        "seed": cfg.seed,
        "does_not_retune_m2": True,
        "does_not_retune_gvis": True,
        "sim_rgb_not_real_camera": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)

    rgb_cfg = _to_rgb_cfg(cfg)
    splits = _collect(rgb_cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    if int(train["rgb"].shape[1]) != int(cfg.rgb_size):
        for p in splits.values():
            p["rgb"] = np.stack([_resize_rgb(im, cfg.rgb_size) for im in p["rgb"]], axis=0)

    n_dof = int(train["n_dof"])
    stats = _state_stats(train["q"], train["qd"])
    ytr = _delta(train)
    w_m2 = _lstsq(_phi_m2(train["q"], train["qd"], train["eq"]), ytr)
    if not cfg.smoke and int(w_m2.size) != P_STRUCT:
        raise RuntimeError(f"M2 param count {w_m2.size} != {P_STRUCT}")
    np.save(root / "W_m2.npy", w_m2)
    scale = float(np.sqrt(np.mean(np.square(_x(train["q"], train["qd"])))))
    b0 = eval_receding_k4(test, _m2_step_fn(w_m2), scale, k=K_EXECUTE)
    g_oracle = _competent(b0)
    print(f"[rtwx-x0rgb1] B0 oracle E_exec={b0['E_exec']:.4f} r_cat={b0['r_cat']}", flush=True)

    p0_tr = _downsample_pool(train, P0_SIZE)
    p0_va = _downsample_pool(val, P0_SIZE)
    p0_te = _downsample_pool(test, P0_SIZE)
    from .rtwx_x0rgb import _make_vis_dataset

    xtr, ytr_v = _make_vis_dataset(p0_tr, L_P0)
    xva, yva_v = _make_vis_dataset(p0_va, L_P0)
    p0_model = _train_vis(
        xtr,
        ytr_v,
        xva,
        yva_v,
        out_dim=int(ytr_v.shape[1]),
        hidden=64 if cfg.smoke else VIS_HIDDEN,
        seed=cfg.seed,
        epochs=5 if cfg.smoke else cfg.vis_epochs,
    )
    p0_perc = _perception_metrics(p0_te, p0_model, L_P0)
    print(f"[rtwx-x0rgb1] P0 E_q={p0_perc['E_q']:.4f} E_qd={p0_perc['E_qd']:.4f}", flush=True)

    ch = (8, 16, 32, 64) if cfg.smoke else cfg.cnn_ch
    epochs = 5 if cfg.smoke else cfg.vis_epochs
    bs = 16 if cfg.smoke else 32

    p1 = _train_split_heads(
        _build_p1(n_dof, ch),
        _iter_p1(train, stats, bs),
        _iter_p1(val, stats, bs),
        seed=cfg.seed,
        epochs=epochs,
    )
    q1, d1 = _predict_p1(p1, test, stats)
    p1_perc = _audit(q1, d1, test)
    print(
        f"[rtwx-x0rgb1] P1 E_q={p1_perc['E_q']:.4f} E_qd={p1_perc['E_qd']:.4f} n_q_ok={p1_perc['n_q_below_035']}",
        flush=True,
    )

    p2 = _train_split_heads(
        _build_p2(n_dof, ch, L_TEMP),
        _iter_p2(train, stats, L_TEMP, max(8, bs // 2)),
        _iter_p2(val, stats, L_TEMP, bs),
        seed=cfg.seed + 1,
        epochs=epochs,
    )
    q2, d2 = _predict_p2(p2, test, stats, L_TEMP)
    p2_perc = _audit(q2, d2, test)
    print(
        f"[rtwx-x0rgb1] P2 E_q={p2_perc['E_q']:.4f} E_qd={p2_perc['E_qd']:.4f} n_q_ok={p2_perc['n_q_below_035']}",
        flush=True,
    )

    g_vis = bool(p2_perc["E_q"] <= E_Q_MAX and p2_perc["E_qd"] <= E_QD_MAX)
    vis_deploy = None
    eta = None
    run_deploy = g_vis or cfg.smoke
    if run_deploy:
        vis_reset = _p2_reset_fn(test, p2, stats, L_TEMP)
        vis_deploy = eval_receding_k4_vis(
            test, _m2_step_fn(w_m2), vis_reset, scale, k=K_EXECUTE, oracle_reset=False
        )
        denom = float(b0["E_identity"] - b0["E_exec"])
        if np.isfinite(denom) and denom > 1.0e-8:
            eta = float((vis_deploy["E_identity"] - vis_deploy["E_exec"]) / denom)
        print(f"[rtwx-x0rgb1] P2→M2 E_exec={vis_deploy['E_exec']:.4f} r_cat={vis_deploy['r_cat']}", flush=True)

    if g_vis:
        pattern = _pattern_p2(p2_perc, vis_deploy)
    else:
        pattern = _pattern_p2(p2_perc, None)

    summary = {
        "header": header,
        "pattern": pattern,
        "G_oracle": g_oracle,
        "G_vis_P2": g_vis,
        "P0_negative_control": p0_perc,
        "P1": p1_perc,
        "P2": p2_perc,
        "B0_oracle": {"test_K4": b0, "P": int(w_m2.size)},
        "B1_P2_to_M2": vis_deploy,
        "eta_retain": eta if g_vis else None,
        "eta_retain_logged": eta,
        "does_not_claim_real_camera": True,
        "capacity_claim": False,
        "sim_rgb_state_interface_claim": pattern == "sim_rgb_state_interface_supported",
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            "pattern": pattern,
            "G_oracle": g_oracle,
            "G_vis_P2": g_vis,
            "P0": p0_perc,
            "P1": p1_perc,
            "P2": p2_perc,
            "B0": b0,
            "B1": vis_deploy,
        },
    )
    return summary
