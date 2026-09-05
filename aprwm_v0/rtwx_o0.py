"""RTWX-O0: RGB → explicit cup state. No M2. RGB↛(q,qd)."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0 import _nrmse
from .rtwx_x0c import FORMAL_N_STEPS, FORMAL_N_TEST, FORMAL_N_TRAIN, FORMAL_N_VAL, _write_json
from .rtwx_x0e1 import LR, PATIENCE, WD
from .rtwx_x0rgb import _capture_rgb_step, _patch_curobo_planner, _patch_raster_shader, _resize_rgb, _rgb_args

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0.object_state_observability.v1"
TASK = "place_empty_cup"
SEED_FORMAL = 16601
RGB_SIZE = 224
L_HIST = 4
E_P_MAX = 0.30
MED_ER_MAX = 15.0
P90_ER_MAX = 30.0
E_V_MAX = 0.50
E_W_MAX = 0.60
INFO_RATIO = 0.85
EPOCHS = 40


@dataclass(frozen=True)
class RTWXO0Config:
    output: str = "runs/rtwx_o0"
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
    rgb_size: int = RGB_SIZE
    vis_epochs: int = EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0 must not write there")


def _lock(cfg: RTWXO0Config) -> RTWXO0Config:
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
        vis_epochs=EPOCHS,
    )


def _quat_fix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1.0e-8, None)
    return np.where(q[:, :1] < 0, -q, q)


def _e_R_deg(qhat: np.ndarray, q: np.ndarray) -> np.ndarray:
    a, b = _quat_fix(qhat), _quat_fix(q)
    d = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(d))


def _cup_pose(env: Any) -> tuple[np.ndarray, np.ndarray]:
    pose = env.cup.get_pose()
    return np.asarray(pose.p, dtype=np.float64).reshape(3), _quat_fix(np.asarray(pose.q, dtype=np.float64))[0]


def _cup_twist(env: Any) -> tuple[np.ndarray, np.ndarray]:
    cup = env.cup
    ent = getattr(cup, "actor", cup)
    if hasattr(ent, "get_components"):
        try:
            for c in ent.get_components():
                lin, ang = getattr(c, "linear_velocity", None), getattr(c, "angular_velocity", None)
                if lin is not None and ang is not None:
                    return np.asarray(lin, dtype=np.float64).reshape(3), np.asarray(ang, dtype=np.float64).reshape(3)
        except Exception:
            pass
    return np.full(3, np.nan), np.full(3, np.nan)


def _fill_twist_fd(p: np.ndarray, quat: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros_like(p)
    w = np.zeros_like(p)
    v[1:] = (p[1:] - p[:-1]) / np.clip(dt[1:, None], 1.0e-4, None)
    v[0] = v[1]
    q = _quat_fix(quat)
    for t in range(1, p.shape[0]):
        w[t] = (q[t, 1:4] - q[t - 1, 1:4]) / max(float(dt[t]), 1.0e-4)
    w[0] = w[1]
    return v, w


def _o0_args(repo: Path) -> dict[str, Any]:
    args = _rgb_args(repo)
    args["task_name"] = TASK
    return args


def _synth_from_p(p: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    feat = np.concatenate([p, np.sin(10 * p), np.cos(10 * p)])
    feat = feat + rng.normal(0, 0.02, size=feat.shape)
    side = int(np.ceil(np.sqrt(feat.size / 3.0)))
    pad = np.zeros(side * side * 3)
    pad[: feat.size] = feat
    img = ((pad.reshape(side, side, 3) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return _resize_rgb(img, size)


def _collect_numpy(cfg: RTWXO0Config, split: str, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    ps, qs, vs, ws, rgbs, dts = [], [], [], [], [], []
    for _ep in range(n_ep):
        p0 = rng.uniform([-0.25, -0.15, 0.02], [0.25, 0.1, 0.08])
        for t in range(n_steps):
            p = p0 + 0.002 * t * rng.normal(size=3)
            quat = _quat_fix(np.array([1.0, 0.05 * np.sin(t / 10.0), 0.0, 0.0]))[0]
            ps.append(p)
            qs.append(quat)
            vs.append(rng.normal(0, 0.01, size=3))
            ws.append(rng.normal(0, 0.02, size=3))
            rgbs.append(_synth_from_p(p, rng, cfg.rgb_size))
            dts.append(0.05)
    return {
        "p": np.stack(ps),
        "quat": np.stack(qs),
        "v": np.stack(vs),
        "omega": np.stack(ws),
        "rgb": np.stack(rgbs),
        "dt": np.asarray(dts, dtype=np.float64),
        "n_ep": n_ep,
        "n_steps": n_steps,
    }


def collect_cup_rgb(cfg: RTWXO0Config, *, split: str, rng: np.random.Generator, stop_mode: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import (
        _arm_q,
        _build_action,
        _contact_invalid,
        _count_steps,
        _drive_pack,
        _joint_limits,
        _multisine,
    )

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
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
            print(f"[rtwx-o0] {split} valid {len(eps)}/{n_ep} attempt {attempts_used}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            if getattr(env, "step_lim", 1000) is None or int(env.step_lim) <= int(cfg.n_steps):
                env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup"):
                reasons.append("no_cup")
                env.close()
                continue
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
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            dt_guess = scene_dt * 50.0
            tt = np.arange(cfg.n_steps, dtype=np.float64) * dt_guess
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            rows: dict[str, list] = {k: [] for k in ("p", "quat", "v", "omega", "rgb", "dt")}
            invalid = None
            for i in range(int(cfg.n_steps)):
                drv_pre = _drive_pack(jl + jr)
                p, quat = _cup_pose(env)
                lin, ang = _cup_twist(env)
                rows["rgb"].append(_capture_rgb_step(env, cfg.rgb_size))
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["v"].append(lin)
                rows["omega"].append(ang)
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig_step, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig_step
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "take_action_noop"
                    break
                n_int = int(box[0])
                rows["dt"].append(max(n_int * scene_dt, scene_dt))
                drv_post = _drive_pack(jl + jr)
                n_l = q0_l.size
                cmd_l = np.asarray(action[:n_l], dtype=np.float64)
                cmd_r = np.asarray(action[n_l + 1 : n_l + 1 + q0_r.size], dtype=np.float64)
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
            try:
                env.close()
            except Exception:
                pass
            if invalid is not None:
                reasons.append(invalid)
                continue
            if len(rows["p"]) != int(cfg.n_steps):
                reasons.append("short_episode")
                continue
            ep = {k: (np.stack(v) if k == "rgb" else np.asarray(v, dtype=np.float64)) for k, v in rows.items()}
            if not np.isfinite(ep["v"]).all():
                ep["v"], ep["omega"] = _fill_twist_fd(ep["p"], ep["quat"], ep["dt"])
            eps.append(ep)
        except Exception as exc:
            reasons.append(f"setup:{type(exc).__name__}:{exc}")
            print(f"[rtwx-o0] {split} fail {attempts_used}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop_mode.append(f"{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0 collect {split}: 0 episodes; {reasons[-20:]}")
    out: dict[str, Any] = {k: np.concatenate([e[k] for e in eps], axis=0) for k in ("p", "quat", "v", "omega", "rgb", "dt")}
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    return out


def _save_split(path: Path, name: str, p: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": p[k] for k in ("p", "quat", "v", "omega", "rgb", "dt")}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    np.savez_compressed(path, **payload)


def _load_split(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in ("p", "quat", "v", "omega", "dt")}
    p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect(cfg: RTWXO0Config, root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(cfg.seed),
        "val": np.random.default_rng(cfg.seed + 1),
        "test": np.random.default_rng(cfg.seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0_{name}.npz"
        hit = _load_split(part, name)
        if hit is not None:
            print(f"[rtwx-o0] {name}: cache", flush=True)
            splits[name] = hit
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
            pool = collect_cup_rgb(cfg, split=name, rng=rng, stop_mode=stop)
        splits[name] = pool
        _save_split(part, name, pool)
    if stop:
        raise RuntimeError(f"O0 collect stop: {stop}")
    return splits


def _clip(rgb: np.ndarray, s0: int, t: int, l: int) -> np.ndarray:
    idx = [s0 + max(0, t - (l - 1 - k)) for k in range(l)]
    x = np.asarray(rgb[idx], dtype=np.float32) / 255.0
    return np.transpose(x, (0, 3, 1, 2))


def _build_encoder(*, n_out: int, l_hist: int, smoke: bool):
    from torch import nn

    if smoke:
        cin, spat = 32, 16
        body = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
        )
    else:
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet18

            net = resnet18(weights=None)
        body = nn.Sequential(*list(net.children())[:-2])
        cin, spat = 512, 49

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            self.proj = nn.Conv2d(32 if smoke else 512, 32 if smoke else 64, 1)
            fd = (32 if smoke else 64) * spat
            hid = 256 if smoke else 512
            self.temp = nn.Sequential(nn.Linear(fd * l_hist, hid), nn.SiLU(), nn.Linear(hid, n_out))

        def forward(self, x):
            b, l, c, h, w = x.shape
            f = self.proj(self.body(x.reshape(b * l, c, h, w))).reshape(b, l, -1)
            return self.temp(f.reshape(b, -1))

    return Enc()


def _train(model: Any, batches: list, val_batches: list, *, seed: int, epochs: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(batches))
        for i in order:
            x, y = batches[int(i)]
            opt.zero_grad()
            pred = model(torch.from_numpy(x).to(dev))
            loss = loss_fn(pred, torch.from_numpy(y).float().to(dev))
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for x, y in val_batches:
                pred = model(torch.from_numpy(x).to(dev))
                vals.append(float(loss_fn(pred, torch.from_numpy(y).float().to(dev)).item()))
        v = float(np.mean(vals)) if vals else float("inf")
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _stats(train: dict[str, Any]) -> dict[str, np.ndarray]:
    def ms(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = a.mean(0), a.std(axis=0)
        return mu, np.where(sd < 1e-8, 1.0, sd)

    mu_p, sd_p = ms(train["p"])
    mu_q, sd_q = ms(train["quat"])
    mu_v, sd_v = ms(train["v"])
    mu_w, sd_w = ms(train["omega"])
    return {"mu_p": mu_p, "sd_p": sd_p, "mu_q": mu_q, "sd_q": sd_q, "mu_v": mu_v, "sd_v": sd_v, "mu_w": mu_w, "sd_w": sd_w}


def _make_xy(pool: dict[str, Any], stats: dict[str, np.ndarray], l_hist: int, bs: int) -> list:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    y = np.concatenate(
        [
            (pool["p"] - stats["mu_p"]) / stats["sd_p"],
            (pool["quat"] - stats["mu_q"]) / stats["sd_q"],
            (pool["v"] - stats["mu_v"]) / stats["sd_v"],
            (pool["omega"] - stats["mu_w"]) / stats["sd_w"],
        ],
        axis=1,
    ).astype(np.float32)
    xs, ys = [], []
    for ep in range(n_ep):
        s0 = ep * n_steps
        for t in range(n_steps):
            xs.append(_clip(pool["rgb"], s0, t, l_hist))
            ys.append(y[s0 + t])
    x, y = np.stack(xs), np.stack(ys)
    return [(x[i : i + bs], y[i : i + bs]) for i in range(0, x.shape[0], bs)]


def _predict(model: Any, pool: dict[str, Any], stats: dict[str, np.ndarray], l_hist: int) -> dict[str, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    rows: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    model.eval()

    def flush() -> None:
        if not buf:
            return
        with torch.no_grad():
            rows.append(model(torch.from_numpy(np.stack(buf)).to(dev)).cpu().numpy())
        buf.clear()

    for ep in range(n_ep):
        s0 = ep * n_steps
        for t in range(n_steps):
            buf.append(_clip(pool["rgb"], s0, t, l_hist))
            if len(buf) >= 16:
                flush()
    flush()
    y = np.concatenate(rows, axis=0)
    return {
        "p": y[:, :3] * stats["sd_p"] + stats["mu_p"],
        "quat": _quat_fix(y[:, 3:7] * stats["sd_q"] + stats["mu_q"]),
        "v": y[:, 7:10] * stats["sd_v"] + stats["mu_v"],
        "omega": y[:, 10:13] * stats["sd_w"] + stats["mu_w"],
    }


def _score(pred: dict[str, np.ndarray], pool: dict[str, Any]) -> dict[str, Any]:
    er = _e_R_deg(pred["quat"], pool["quat"])
    ep_cm = np.linalg.norm(pred["p"] - pool["p"], axis=1) * 100.0
    return {
        "E_p": _nrmse(pred["p"], pool["p"]),
        "median_ep_cm": float(np.median(ep_cm)),
        "mean_ep_cm": float(np.mean(ep_cm)),
        "median_e_R_deg": float(np.median(er)),
        "p90_e_R_deg": float(np.percentile(er, 90)),
        "E_v": _nrmse(pred["v"], pool["v"]),
        "E_omega": _nrmse(pred["omega"], pool["omega"]),
    }


def _b0_pred(train: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    return {
        "p": np.repeat(train["p"].mean(0, keepdims=True), n, 0),
        "quat": np.repeat(_quat_fix(train["quat"].mean(0, keepdims=True)), n, 0),
        "v": np.repeat(train["v"].mean(0, keepdims=True), n, 0),
        "omega": np.repeat(train["omega"].mean(0, keepdims=True), n, 0),
    }


def _b1_prev(pool: dict[str, Any]) -> dict[str, np.ndarray]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    p, q, v, w = pool["p"].copy(), pool["quat"].copy(), pool["v"].copy(), pool["omega"].copy()
    for ep in range(n_ep):
        s0 = ep * n_steps
        p[s0 + 1 : s0 + n_steps] = pool["p"][s0 : s0 + n_steps - 1]
        q[s0 + 1 : s0 + n_steps] = pool["quat"][s0 : s0 + n_steps - 1]
        v[s0 + 1 : s0 + n_steps] = pool["v"][s0 : s0 + n_steps - 1]
        w[s0 + 1 : s0 + n_steps] = pool["omega"][s0 : s0 + n_steps - 1]
    return {"p": p, "quat": q, "v": v, "omega": w}


def run_rtwx_o0(output: str | Path, config: RTWXO0Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "task": TASK,
        "closes_rgb_to_robot_q": True,
        "no_m2": True,
        "seed": cfg.seed,
        "rgb_size": cfg.rgb_size,
        "L": L_HIST,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    splits = _collect(cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    n = int(test["p"].shape[0])
    b0 = _score(_b0_pred(train, n), test)
    b1 = _score(_b1_prev(test), test)
    print(f"[rtwx-o0] B0 E_p={b0['E_p']:.4f} med_eR={b0['median_e_R_deg']:.2f}", flush=True)
    print(f"[rtwx-o0] B1 E_p={b1['E_p']:.4f} med_eR={b1['median_e_R_deg']:.2f}", flush=True)
    stats = _stats(train)
    bs = 8 if cfg.smoke else 12
    model = _train(
        _build_encoder(n_out=13, l_hist=L_HIST, smoke=cfg.smoke),
        _make_xy(train, stats, L_HIST, bs),
        _make_xy(val, stats, L_HIST, bs),
        seed=cfg.seed,
        epochs=4 if cfg.smoke else cfg.vis_epochs,
    )
    b2 = _score(_predict(model, test, stats, L_HIST), test)
    print(
        f"[rtwx-o0] B2 E_p={b2['E_p']:.4f} med_cm={b2['median_ep_cm']:.2f} med_eR={b2['median_e_R_deg']:.2f}",
        flush=True,
    )
    g0 = np.isfinite(test["p"]).all() and (
        cfg.smoke or float(np.std(train["p"][:, 0]) + np.std(train["p"][:, 1])) > 0.02
    )
    g1 = bool(b2["E_p"] <= E_P_MAX)
    g2 = bool(b2["median_e_R_deg"] <= MED_ER_MAX and b2["p90_e_R_deg"] <= P90_ER_MAX)
    g3 = bool(b2["E_v"] <= E_V_MAX and b2["E_omega"] <= E_W_MAX)
    g_info = bool(
        b2["E_p"] <= INFO_RATIO * b0["E_p"]
        and b2["median_e_R_deg"] <= INFO_RATIO * max(b0["median_e_R_deg"], 1e-6)
    )
    if not g0 or not g1 or not g2 or not g_info:
        pattern = "object_pose_failure"
    elif not g3:
        pattern = "pose_supported_velocity_failed"
    else:
        pattern = "explicit_object_state_supported"
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": g0,
        "G1": g1,
        "G2": g2,
        "G3": g3,
        "G_info": g_info,
        "B0_mean": b0,
        "B1_prev_gt": b1,
        "B2_rgb": b2,
        "no_m2": True,
        "rgb_to_robot_q_closed": True,
        "does_not_claim_real_camera": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {k: summary[k] for k in ("pattern", "G0", "G1", "G2", "G3", "G_info", "B0_mean", "B1_prev_gt", "B2_rgb")},
    )
    return summary


