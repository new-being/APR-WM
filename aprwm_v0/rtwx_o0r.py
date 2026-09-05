"""RTWX-O0R: fresh RGB→s^O with frozen head_camera+CoordConv. No arch search."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import (
    TASK,
    _cup_pose,
    _cup_twist,
    _e_R_deg,
    _fill_twist_fd,
    _o0_args,
    _quat_fix,
)
from .rtwx_o0d import _blob_frame, _cup_ids
from .rtwx_o0d1 import _project
from .rtwx_o0d3 import (
    _capture_head,
    _coord_net,
    _e_R_after_yaw,
    _head_cam,
    _nchw,
)
from .rtwx_x0 import _nrmse
from .rtwx_x0c import FORMAL_N_STEPS, FORMAL_N_TEST, FORMAL_N_TRAIN, FORMAL_N_VAL, _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0R_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0r.fresh_object_state_observability.v1"
CAMERA = "head_camera"
SEED = 21601
RGB_SIZE = 64
E_P_MAX = 0.30
INFO_RATIO = 0.85
MED_ER_MAX = 15.0
P90_ER_MAX = 30.0
E_V_MAX = 0.50
E_W_MAX = 0.60
G0_FOV = 0.95
G0_VIS = 0.90
MASK_PX = 50
SYM_MED = 5.0
SYM_P90 = 15.0
EPOCHS = 40
PATIENCE = 8
LR = 1.0e-3


@dataclass(frozen=True)
class RTWXO0RConfig:
    output: str = "runs/rtwx_o0r"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = FORMAL_N_TRAIN
    n_val_ep: int = FORMAL_N_VAL
    n_test_ep: int = FORMAL_N_TEST
    n_steps: int = FORMAL_N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    vis_epochs: int = EPOCHS


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0R must not write there")


def _lock(cfg: RTWXO0RConfig) -> RTWXO0RConfig:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=FORMAL_N_TRAIN,
        n_val_ep=FORMAL_N_VAL,
        n_test_ep=FORMAL_N_TEST,
        n_steps=FORMAL_N_STEPS,
        seed=SEED,
        rgb_size=RGB_SIZE,
        vis_epochs=EPOCHS,
    )


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


def _b0(train: dict[str, Any], n: int) -> dict[str, np.ndarray]:
    return {
        "p": np.repeat(train["p"].mean(0, keepdims=True), n, 0),
        "quat": np.repeat(_quat_fix(train["quat"].mean(0, keepdims=True)), n, 0),
        "v": np.repeat(train["v"].mean(0, keepdims=True), n, 0),
        "omega": np.repeat(train["omega"].mean(0, keepdims=True), n, 0),
    }


def _b1(pool: dict[str, Any]) -> dict[str, np.ndarray]:
    n_ep, n_steps = int(pool["n_ep"]), int(pool["n_steps"])
    p, q, v, w = pool["p"].copy(), pool["quat"].copy(), pool["v"].copy(), pool["omega"].copy()
    for ep in range(n_ep):
        s0 = ep * n_steps
        p[s0 + 1 : s0 + n_steps] = pool["p"][s0 : s0 + n_steps - 1]
        q[s0 + 1 : s0 + n_steps] = pool["quat"][s0 : s0 + n_steps - 1]
        v[s0 + 1 : s0 + n_steps] = pool["v"][s0 : s0 + n_steps - 1]
        w[s0 + 1 : s0 + n_steps] = pool["omega"][s0 : s0 + n_steps - 1]
    return {"p": p, "quat": q, "v": v, "omega": w}


def _stats(train: dict[str, Any]) -> dict[str, np.ndarray]:
    def ms(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = a.mean(0), a.std(0)
        return mu, np.where(sd < 1e-8, 1.0, sd)

    out = {}
    for k in ("p", "quat", "v", "omega"):
        mu, sd = ms(train[k])
        out[f"mu_{k}"], out[f"sd_{k}"] = mu, sd
    return out


def _y(pool: dict[str, Any], st: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            (pool["p"] - st["mu_p"]) / st["sd_p"],
            (pool["quat"] - st["mu_quat"]) / st["sd_quat"],
            (pool["v"] - st["mu_v"]) / st["sd_v"],
            (pool["omega"] - st["mu_omega"]) / st["sd_omega"],
        ],
        axis=1,
    ).astype(np.float32)


def _batches(rgb: np.ndarray, y: np.ndarray, bs: int) -> list:
    x = _nchw(rgb)
    return [(x[i : i + bs], y[i : i + bs]) for i in range(0, x.shape[0], bs)]


def _train(model: Any, tr: list, va: list, *, seed: int, epochs: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(tr))
        for i in order:
            xb, yb = tr[int(i)]
            opt.zero_grad()
            loss_fn(model(torch.from_numpy(xb).to(dev)), torch.from_numpy(yb).float().to(dev)).backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va:
                vals.append(float(loss_fn(model(torch.from_numpy(xb).to(dev)), torch.from_numpy(yb).float().to(dev))))
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


def _predict(model: Any, rgb: np.ndarray, st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    dev = next(model.parameters()).device
    x = _nchw(rgb)
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            rows.append(model(torch.from_numpy(x[i : i + 16]).to(dev)).cpu().numpy())
    y = np.concatenate(rows, axis=0)
    return {
        "p": y[:, :3] * st["sd_p"] + st["mu_p"],
        "quat": _quat_fix(y[:, 3:7] * st["sd_quat"] + st["mu_quat"]),
        "v": y[:, 7:10] * st["sd_v"] + st["mu_v"],
        "omega": y[:, 10:13] * st["sd_omega"] + st["mu_omega"],
    }


def _numpy_split(cfg: RTWXO0RConfig, split: str, rng: np.random.Generator) -> dict[str, Any]:
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    n_steps = int(cfg.n_steps)
    ps, qs, vs, ws, rgbs, fov, vis = [], [], [], [], [], [], []
    for ep in range(n_ep):
        p0 = np.array([-0.25 + 0.5 * (ep / max(n_ep - 1, 1)), rng.uniform(-0.15, 0.05), 0.71])
        for t in range(n_steps):
            p = p0 + np.array([0.001 * np.sin(t / 10.0), 0.0, 0.0])
            quat = _quat_fix(np.array([1.0, 0.02 * np.sin(t / 8.0 + ep), 0.0, 0.0]))[0]
            rgb, mask = _blob_frame(p, cfg.rgb_size)
            rgb = rgb.copy()
            rgb[0, (ep * 3 + t) % 64, 2] = 40 + ((ep + t) % 200)
            ps.append(p)
            qs.append(quat)
            vs.append(np.array([0.001 * np.cos(t / 10.0), 0.0, 0.0]))
            ws.append(np.array([0.0, 0.02 * np.cos(t / 8.0 + ep), 0.0]))
            rgbs.append(rgb)
            fov.append(True)
            vis.append(True)
    return {
        "p": np.stack(ps),
        "quat": np.stack(qs),
        "v": np.stack(vs),
        "omega": np.stack(ws),
        "rgb": np.stack(rgbs),
        "in_fov": np.asarray(fov),
        "visible": np.asarray(vis),
        "n_ep": n_ep,
        "n_steps": n_steps,
    }


def collect_split(cfg: RTWXO0RConfig, *, split: str, rng: np.random.Generator, stop: list[str]) -> dict[str, Any]:
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    _patch_raster_shader()
    args = _o0_args(repo)
    n_ep = {"train": cfg.n_train_ep, "val": cfg.n_val_ep, "test": cfg.n_test_ep}[split]
    seed0 = cfg.seed + {"train": 0, "val": 10_000, "test": 20_000}[split]
    eps: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(eps) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(seed0 + slot * 17 + attempts)
        slot += 1
        if len(eps) % 2 == 0 or len(eps) + 1 == n_ep:
            print(f"[rtwx-o0r] {split} valid {len(eps)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            sim_ids = _cup_ids(env)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            if hasattr(env.scene, "timestep"):
                try:
                    scene_dt = float(env.scene.timestep)
                except Exception:
                    pass
            tt = np.arange(cfg.n_steps) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            cam = _head_cam(env)
            rows: dict[str, list] = {k: [] for k in ("p", "quat", "v", "omega", "rgb", "dt", "in_fov", "visible")}
            invalid = None
            repaired = None
            for i in range(int(cfg.n_steps)):
                p, quat = _cup_pose(env)
                lin, ang = _cup_twist(env)
                rgb = _capture_head(env, cfg.rgb_size)
                try:
                    seg = np.asarray(cam.get_picture("Segmentation"))
                    actor = np.asarray(seg[..., 1]).astype(np.int32)
                except Exception:
                    actor = np.zeros((480, 640), dtype=np.int32)
                h, w = actor.shape[:2]
                K = np.asarray(cam.get_intrinsic_matrix())
                E = np.asarray(cam.get_extrinsic_matrix())
                pr = _project(K, E, p, w, h)
                uu = int(np.clip(round(pr["u"]), 0, w - 1))
                vv = int(np.clip(round(pr["v"]), 0, h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                if repaired is None and pr["in_fov"] and id_uv >= 0:
                    repaired = id_uv
                cid = repaired if repaired is not None else (sorted(sim_ids)[0] if sim_ids else -1)
                area = int((actor == cid).sum()) if cid >= 0 else 0
                in_fov = bool(pr["in_fov"])
                visible = bool(in_fov and (area >= MASK_PX or id_uv >= 0))
                rows["rgb"].append(rgb)
                rows["p"].append(p)
                rows["quat"].append(quat)
                rows["v"].append(lin)
                rows["omega"].append(ang)
                rows["in_fov"].append(in_fov)
                rows["visible"].append(visible)
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "noop"
                    break
                rows["dt"].append(max(int(box[0]) * scene_dt, scene_dt))
                if _contact_invalid(env):
                    invalid = "contact"
                    break
            env.close()
            if invalid or len(rows["p"]) != int(cfg.n_steps):
                continue
            ep = {
                "p": np.asarray(rows["p"], dtype=np.float64),
                "quat": np.asarray(rows["quat"], dtype=np.float64),
                "v": np.asarray(rows["v"], dtype=np.float64),
                "omega": np.asarray(rows["omega"], dtype=np.float64),
                "rgb": np.stack(rows["rgb"]),
                "dt": np.asarray(rows["dt"], dtype=np.float64),
                "in_fov": np.asarray(rows["in_fov"], dtype=bool),
                "visible": np.asarray(rows["visible"], dtype=bool),
            }
            if not np.isfinite(ep["v"]).all():
                ep["v"], ep["omega"] = _fill_twist_fd(ep["p"], ep["quat"], ep["dt"])
            eps.append(ep)
        except Exception as exc:
            print(f"[rtwx-o0r] {split} fail {attempts}: {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(eps) < n_ep:
        stop.append(f"{split}:only_{len(eps)}_of_{n_ep}")
    if not eps:
        raise RuntimeError(f"O0R collect {split}: 0 episodes")
    out: dict[str, Any] = {
        k: np.concatenate([e[k] for e in eps], axis=0)
        for k in ("p", "quat", "v", "omega", "rgb", "in_fov", "visible")
    }
    out["n_ep"] = len(eps)
    out["n_steps"] = int(cfg.n_steps)
    return out


def _save(path: Path, name: str, p: dict[str, Any]) -> None:
    payload = {f"{name}_{k}": p[k] for k in ("p", "quat", "v", "omega", "rgb", "in_fov", "visible")}
    payload[f"{name}_n_ep"] = np.array([p["n_ep"]])
    payload[f"{name}_n_steps"] = np.array([p["n_steps"]])
    np.savez_compressed(path, **payload)


def _load(path: Path, name: str) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    z = np.load(path)
    if f"{name}_p" not in z:
        return None
    p = {k: np.asarray(z[f"{name}_{k}"]) for k in ("p", "quat", "v", "omega", "in_fov", "visible")}
    p["rgb"] = np.asarray(z[f"{name}_rgb"], dtype=np.uint8)
    p["n_ep"] = int(z[f"{name}_n_ep"][0])
    p["n_steps"] = int(z[f"{name}_n_steps"][0])
    return p


def _collect(cfg: RTWXO0RConfig, root: Path) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    stop: list[str] = []
    rngs = {
        "train": np.random.default_rng(cfg.seed),
        "val": np.random.default_rng(cfg.seed + 1),
        "test": np.random.default_rng(cfg.seed + 2),
    }
    for name, rng in rngs.items():
        part = root / f"cache_o0r_{name}.npz"
        hit = _load(part, name)
        if hit is not None:
            print(f"[rtwx-o0r] {name}: cache", flush=True)
            splits[name] = hit
            continue
        if cfg.backend == "numpy":
            pool = _numpy_split(cfg, name, rng)
        else:
            repo = Path(cfg.robotwin_repo)
            os.chdir(repo)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
            _patch_curobo_planner(repo)
            pool = collect_split(cfg, split=name, rng=rng, stop=stop)
        splits[name] = pool
        _save(part, name, pool)
    if stop:
        raise RuntimeError(f"O0R collect stop: {stop}")
    return splits


def _pattern(*, g0: bool, g_pos: bool, g_info: bool, g_ori: bool, g_vel: bool, sym: bool) -> str:
    if not g0:
        return "coverage_failure"
    if not g_pos or not g_info:
        return "object_pose_failure"
    if not g_ori:
        return "orientation_symmetry_limited" if sym else "object_pose_failure"
    if not g_vel:
        return "pose_supported_velocity_failed"
    return "explicit_object_pose_supported"


def run_rtwx_o0r(output: str | Path, config: RTWXO0RConfig | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0RConfig())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0R",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "camera": CAMERA,
        "rgb_size": RGB_SIZE,
        "encoder": "CoordConv+spatial_flatten_no_GAP",
        "depends_on": "O0D3=instrument_qualified",
        "seed": cfg.seed,
        "no_m2": True,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    splits = _collect(cfg, root)
    train, val, test = splits["train"], splits["val"], splits["test"]
    n = int(test["p"].shape[0])
    p_fov = float(np.mean(test["in_fov"]))
    p_vis = float(np.mean(test["visible"]))
    g0 = bool(p_fov >= G0_FOV and p_vis >= G0_VIS)
    print(f"[rtwx-o0r] G0 FOV={p_fov:.3f} vis={p_vis:.3f} ok={g0}", flush=True)

    b0 = _score(_b0(train, n), test)
    b1 = _score(_b1(test), test)
    print(f"[rtwx-o0r] B0 E_p={b0['E_p']:.4f} med_eR={b0['median_e_R_deg']:.2f}", flush=True)
    print(f"[rtwx-o0r] B1 E_p={b1['E_p']:.4f} med_eR={b1['median_e_R_deg']:.2f}", flush=True)

    st = _stats(train)
    bs = 8 if cfg.smoke else 12
    model = _train(
        _coord_net(13, cfg.rgb_size),
        _batches(train["rgb"], _y(train, st), bs),
        _batches(val["rgb"], _y(val, st), bs),
        seed=cfg.seed,
        epochs=4 if cfg.smoke else cfg.vis_epochs,
    )
    pred = _predict(model, test["rgb"], st)
    b2 = _score(pred, test)
    print(
        f"[rtwx-o0r] B2 E_p={b2['E_p']:.4f} med_cm={b2['median_ep_cm']:.2f} med_eR={b2['median_e_R_deg']:.2f}",
        flush=True,
    )

    g_pos = bool(b2["E_p"] <= E_P_MAX)
    g_info = bool(b2["E_p"] <= INFO_RATIO * b0["E_p"])
    g_ori = bool(b2["median_e_R_deg"] <= MED_ER_MAX and b2["p90_e_R_deg"] <= P90_ER_MAX)
    g_vel = bool(b2["E_v"] <= E_V_MAX and b2["E_omega"] <= E_W_MAX)
    er_yaw = _e_R_after_yaw(pred["quat"], test["quat"])
    med_y, p90_y = float(np.median(er_yaw)), float(np.percentile(er_yaw, 90))
    sym = bool((not g_ori) and med_y < SYM_MED and p90_y < SYM_P90)
    pattern = _pattern(g0=g0, g_pos=g_pos, g_info=g_info, g_ori=g_ori, g_vel=g_vel, sym=sym)
    unlock_o1 = pattern in {"explicit_object_pose_supported", "pose_supported_velocity_failed"}
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": g0, "P_FOV": p_fov, "P_visible": p_vis},
        "G_pos": g_pos,
        "G_info": g_info,
        "G_ori": g_ori,
        "G_vel": g_vel,
        "B0_mean": b0,
        "B1_prev_gt": b1,
        "B2_rgb": b2,
        "symmetry_secondary": {
            "median_e_R_yaw": med_y,
            "p90_e_R_yaw": p90_y,
            "symmetry_limited": sym,
            "does_not_override_primary": True,
        },
        "unlocks_o1": unlock_o1,
        "no_m2": True,
        "does_not_claim_real_camera": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "metrics.json",
        {
            k: summary[k]
            for k in (
                "pattern",
                "G0",
                "G_pos",
                "G_info",
                "G_ori",
                "G_vel",
                "B0_mean",
                "B1_prev_gt",
                "B2_rgb",
                "symmetry_secondary",
                "unlocks_o1",
            )
        },
    )
    return summary
