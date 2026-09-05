"""RTWX-O0D1: perception instrument repair. No O0R/O1. No scientific claim."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _o0_args
from .rtwx_o0d import (
    _blob_frame,
    _cup_ids,
    _front_sapien_cam,
    _load_o0_train,
    _resize_mask,
)
from .rtwx_x0 import _nrmse
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import CAMERA, _capture_rgb_step, _patch_curobo_planner, _patch_raster_shader

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0D1_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0d1.perception_instrument_repair.v1"
SEED = 18601
N_EP = 6
N_STEPS = 20
N_SCALAR = 32
N_POSE = 256
R1_MAX = 0.05
R2_MAX = 0.10
R2_MEAN_RATIO = 0.20
R3_MIN = 0.90
MASK_PX_MIN = 50
EPOCH_SCALAR = 400
EPOCH_POSE = 200
EPOCH_RAND = 400


@dataclass(frozen=True)
class RTWXO0D1Config:
    output: str = "runs/rtwx_o0d1"
    o0_cache: str = "runs/rtwx_o0"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = N_EP
    n_steps: int = N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = 224
    n_scalar: int = N_SCALAR
    n_pose: int = N_POSE
    epoch_scalar: int = EPOCH_SCALAR
    epoch_pose: int = EPOCH_POSE


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0D1 must not write there")


def _lock(cfg: RTWXO0D1Config) -> RTWXO0D1Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=N_EP,
        n_steps=N_STEPS,
        seed=SEED,
        rgb_size=224,
        n_scalar=N_SCALAR,
        n_pose=N_POSE,
        epoch_scalar=EPOCH_SCALAR,
        epoch_pose=EPOCH_POSE,
    )


def denorm_smoke(mu: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    z = np.ones_like(mu)
    hat = z * sd + mu
    ok = bool(np.allclose(hat, mu + sd))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "hat_from_z1": hat.tolist(), "ok": ok}


def _actor_ids(obj: Any) -> set[int]:
    ids: set[int] = set()
    if obj is None:
        return ids
    for a in (obj, getattr(obj, "entity", None), getattr(obj, "actor", None)):
        if a is None:
            continue
        if hasattr(a, "get_id"):
            try:
                ids.add(int(a.get_id()))
            except Exception:
                pass
        for attr in ("per_scene_id", "id"):
            if hasattr(a, attr):
                try:
                    ids.add(int(getattr(a, attr)))
                except Exception:
                    pass
    return ids


def _name_of(obj: Any) -> str:
    for a in (obj, getattr(obj, "entity", None)):
        if a is None:
            continue
        if hasattr(a, "get_name"):
            try:
                return str(a.get_name())
            except Exception:
                pass
        if hasattr(a, "name"):
            return str(a.name)
    return "?"


def _scene_id_table(env: Any) -> list[dict[str, Any]]:
    rows = []
    try:
        actors = list(env.scene.get_all_actors())
    except Exception:
        actors = []
    for a in actors:
        rows.append({"name": _name_of(a), "ids": sorted(_actor_ids(a))})
    return rows


def _project(K: np.ndarray, E: np.ndarray, p: np.ndarray, w: int, h: int) -> dict[str, Any]:
    ph = np.array([float(p[0]), float(p[1]), float(p[2]), 1.0], dtype=np.float64)
    e = np.asarray(E, dtype=np.float64)
    if e.shape == (4, 4):
        pc = e @ ph
    else:
        pc = e @ ph[: e.shape[1]]
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    k = np.asarray(K, dtype=np.float64)
    u = float(k[0, 0] * x / max(z, 1e-8) + k[0, 2])
    v = float(k[1, 1] * y / max(z, 1e-8) + k[1, 2])
    in_fov = bool(z > 1e-4 and 0.0 <= u < float(w) and 0.0 <= v < float(h))
    return {"u": u, "v": v, "z": z, "in_fov": in_fov}


def _tiny(n_out: int, size: int):
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 16, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_out),
            )

        def forward(self, x):
            return self.net(x)

    return Tiny()


def _to_nchw(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    if x.ndim == 3:
        return np.transpose(x, (2, 0, 1))
    return np.transpose(x, (0, 3, 1, 2))


def _train_scalar(
    rgb: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(_to_nchw(rgb)).to(dev)
    t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1)).to(dev)
    model = _tiny(1, int(rgb.shape[1])).to(dev)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    n_param = sum(int(p.numel()) for p in model.parameters())
    n_grad = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    theta0 = torch.cat([p.detach().flatten() for p in model.parameters()])
    g_hist, d_hist, loss_hist = [], [], []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, t)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        g_hist.append(float(torch.linalg.norm(g)))
        opt.step()
        theta = torch.cat([p.detach().flatten() for p in model.parameters()])
        d_hist.append(float(torch.linalg.norm(theta - theta0)))
        loss_hist.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        hat = model(x).cpu().numpy().reshape(-1)
    theta1 = torch.cat([p.detach().flatten() for p in model.parameters()])
    return {
        "hat": hat,
        "n_param": n_param,
        "n_grad": n_grad,
        "grad_norm_mean": float(np.mean(g_hist[:20])),
        "grad_norm_last": float(g_hist[-1]),
        "dtheta_from_init": float(torch.linalg.norm(theta1 - theta0)),
        "loss0": float(loss_hist[0]),
        "loss_last": float(loss_hist[-1]),
        "train_mode": True,
        "images_unique_enough": bool(np.std(_to_nchw(rgb).reshape(len(rgb), -1), axis=0).mean() > 1e-4),
    }


def _image_audit(rgb: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(rgb, dtype=np.float32)
    stds = arr.reshape(arr.shape[0], -1).std(axis=1)
    n = min(12, arr.shape[0])
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(float(np.linalg.norm(arr[i].astype(np.float32) - arr[j].astype(np.float32))))
    return {
        "median_pixel_std": float(np.median(stds)),
        "min_pixel_std": float(np.min(stds)),
        "mean_pairwise_l2": float(np.mean(diffs)) if diffs else 0.0,
        "n": int(rgb.shape[0]),
        "not_constant": bool(np.median(stds) > 1.0),
    }


def _overlay(rgb: np.ndarray, mask: np.ndarray, u: float, v: float, info: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    img = np.asarray(rgb, dtype=np.uint8).copy()
    m = np.asarray(mask, dtype=bool)
    if m.shape[:2] == img.shape[:2]:
        img[m] = (0.5 * img[m] + np.array([0, 160, 0])).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    uu, vv = int(round(u)), int(round(v))
    dr.line((uu - 8, vv, uu + 8, vv), fill=(255, 40, 40), width=2)
    dr.line((uu, vv - 8, uu, vv + 8), fill=(255, 40, 40), width=2)
    dr.text((4, 4), info[:80], fill=(255, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _numpy_vis(cfg: RTWXO0D1Config, rng: np.random.Generator) -> dict[str, Any]:
    n = int(cfg.n_train_ep) * int(cfg.n_steps)
    ps, rgbs, masks, vis_flags, in_fovs = [], [], [], [], []
    for _ in range(n):
        p = rng.uniform([-0.3, -0.2, 0.70], [0.3, 0.05, 0.72])
        rgb, mask = _blob_frame(p, cfg.rgb_size)
        ps.append(p)
        rgbs.append(rgb)
        masks.append(mask)
        vis_flags.append(True)
        in_fovs.append(True)
    return {
        "p": np.stack(ps),
        "rgb": np.stack(rgbs),
        "mask_repaired": np.stack(masks),
        "in_fov": np.asarray(in_fovs),
        "visible": np.asarray(vis_flags),
        "id_match_rate": 1.0,
        "id_table": [{"name": "blob", "ids": [1]}],
        "frac_id_at_uv_equals_sim": 1.0,
        "frac_in_fov": 1.0,
        "frac_occluded_in_fov": 0.0,
        "n": n,
    }


def collect_vis(cfg: RTWXO0D1Config, root: Path) -> dict[str, Any]:
    if cfg.backend == "numpy":
        return _numpy_vis(cfg, np.random.default_rng(cfg.seed))
    from .rtwx_x0 import _setup_env as setup
    from .rtwx_x0c import _arm_q, _build_action, _contact_invalid, _count_steps, _drive_pack, _joint_limits, _multisine

    repo = Path(cfg.robotwin_repo)
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    _patch_raster_shader()
    _patch_curobo_planner(repo)
    args = _o0_args(repo)
    rng = np.random.default_rng(cfg.seed)
    n_ep, n_steps = int(cfg.n_train_ep), int(cfg.n_steps)
    recs: list[dict[str, Any]] = []
    id_table: list[dict[str, Any]] | None = None
    attempts, slot = 0, 0
    while len(recs) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(cfg.seed + slot * 17 + attempts)
        slot += 1
        print(f"[rtwx-o0d1] vis {len(recs)}/{n_ep} attempt {attempts}", flush=True)
        env = None
        try:
            env = setup(repo, TASK, seed, cfg.seed_attempts, args)
            env.check_success = lambda *a, **k: False
            env.step_lim = max(int(cfg.n_steps) + 50, 1000)
            if not hasattr(env, "cup") or _contact_invalid(env):
                env.close()
                continue
            if id_table is None:
                id_table = _scene_id_table(env)
            sim_ids = _cup_ids(env)
            jl = list(env.robot.left_arm_joints)
            jr = list(env.robot.right_arm_joints)
            q0_l, q0_r = _arm_q(env, "left"), _arm_q(env, "right")
            lo_l, hi_l = _joint_limits(jl, q0_l)
            lo_r, hi_r = _joint_limits(jr, q0_r)
            scene_dt = 1.0 / 250.0
            tt = np.arange(n_steps) * (scene_dt * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            cam = _front_sapien_cam(env)
            frames = []
            invalid = None
            for i in range(n_steps):
                p, _q = _cup_pose(env)
                rgb = _capture_rgb_step(env, cfg.rgb_size)
                try:
                    seg = np.asarray(cam.get_picture("Segmentation"))
                    actor = np.asarray(seg[..., 1]).astype(np.int32)
                except Exception:
                    actor = np.zeros(rgb.shape[:2], dtype=np.int32)
                native_h, native_w = actor.shape[:2]
                cfgk = env.cameras.get_config().get(CAMERA) or env.cameras.get_config().get("head_camera")
                K = np.asarray(cfgk["intrinsic_cv"]) if cfgk else np.eye(3)
                E = np.asarray(cfgk["extrinsic_cv"]) if cfgk else np.eye(4)[:3]
                pr = _project(K, E, p, native_w, native_h)
                uu = int(np.clip(round(pr["u"]), 0, native_w - 1))
                vv = int(np.clip(round(pr["v"]), 0, native_h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                # repair: id at projection on first in-FOV frame of episode
                frames.append(
                    {
                        "p": p,
                        "rgb": rgb,
                        "actor": actor,
                        "pr": pr,
                        "id_uv": id_uv,
                        "sim_ids": sorted(sim_ids),
                    }
                )
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig_step, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig_step
                if int(getattr(env, "take_action_cnt", 0)) == cnt0:
                    invalid = "noop"
                    break
                if _contact_invalid(env):
                    invalid = "contact"
                    break
            env.close()
            if invalid or len(frames) != n_steps:
                continue
            recs.append({"frames": frames, "sim_ids": sorted(sim_ids)})
        except Exception as exc:
            print(f"[rtwx-o0d1] fail {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(recs) < n_ep:
        raise RuntimeError(f"O0D1 vis collect {len(recs)}/{n_ep}")
    return _pack_vis(recs, cfg, id_table or [], root)


def _pack_vis(recs: list[dict[str, Any]], cfg: RTWXO0D1Config, id_table: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    ov = root / "overlays"
    ov.mkdir(exist_ok=True)
    ps, rgbs, masks, vis, fov, match, occ = [], [], [], [], [], [], []
    k_ov = 0
    id_eq = []
    for rec in recs:
        repaired = None
        for fr in rec["frames"]:
            if fr["pr"]["in_fov"] and fr["id_uv"] >= 0:
                repaired = int(fr["id_uv"])
                break
        for fr in rec["frames"]:
            actor = fr["actor"]
            pr = fr["pr"]
            scale_u = cfg.rgb_size / actor.shape[1]
            scale_v = cfg.rgb_size / actor.shape[0]
            u224 = pr["u"] * scale_u
            v224 = pr["v"] * scale_v
            cup_id = repaired if repaired is not None else (fr["sim_ids"][0] if fr["sim_ids"] else -1)
            m_nat = actor == cup_id if cup_id >= 0 else np.zeros_like(actor, dtype=bool)
            m224 = _resize_mask(m_nat, cfg.rgb_size)
            area = int(m_nat.sum())
            in_fov = bool(pr["in_fov"])
            visible = bool(in_fov and (area >= MASK_PX_MIN or fr["id_uv"] >= 0))
            id_match = bool(fr["id_uv"] in fr["sim_ids"]) if in_fov else False
            occluded = bool(in_fov and area < MASK_PX_MIN)
            ps.append(fr["p"])
            rgbs.append(fr["rgb"])
            masks.append(m224)
            vis.append(visible)
            fov.append(in_fov)
            match.append(id_match)
            occ.append(occluded)
            id_eq.append(id_match)
            if k_ov < 32:
                _overlay(
                    fr["rgb"],
                    m224,
                    u224,
                    v224,
                    f"fov={int(in_fov)} id_uv={fr['id_uv']} sim={fr['sim_ids'][:4]} p={fr['p'][0]:.2f},{fr['p'][1]:.2f}",
                    ov / f"overlay_{k_ov:02d}.png",
                )
                k_ov += 1
    vis_a = np.asarray(vis)
    return {
        "p": np.stack(ps),
        "rgb": np.stack(rgbs),
        "mask_repaired": np.stack(masks),
        "in_fov": np.asarray(fov),
        "visible": vis_a,
        "id_match_rate": float(np.mean(match)),
        "id_table": id_table,
        "frac_id_at_uv_equals_sim": float(np.mean(id_eq)),
        "frac_in_fov": float(np.mean(fov)),
        "frac_occluded_in_fov": float(np.mean(occ)),
        "n": int(len(ps)),
        "P_visible": float(np.mean(vis_a)),
    }


def _pattern(r1: bool, r2: bool, r3: bool) -> str:
    if not r1:
        return "scalar_memorization_failure"
    if not r2:
        return "pose_memorization_failure"
    if not r3:
        return "visibility_contract_failure"
    return "instrument_repair_supported"


def run_rtwx_o0d1(output: str | Path, config: RTWXO0D1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0D1Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0D1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "no_scientific_claim": True,
        "does_not_unlock_o1": True,
        "does_not_run_o0r": True,
        "o0_frozen": "object_pose_failure",
        "o0d_frozen": "perception_instrument_failure",
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    vis = collect_vis(cfg, root)
    if "P_visible" not in vis:
        vis["P_visible"] = float(np.mean(vis["visible"]))
    r3 = bool(vis["P_visible"] >= R3_MIN)
    print(f"[rtwx-o0d1] R3 P_visible={vis['P_visible']:.3f} in_fov={vis.get('frac_in_fov')} id_match={vis.get('frac_id_at_uv_equals_sim')}", flush=True)

    o0 = None if cfg.smoke else _load_o0_train(Path(cfg.o0_cache))
    src_rgb = vis["rgb"] if o0 is None else o0["rgb"]
    src_p = vis["p"] if o0 is None else o0["p"]
    img = _image_audit(src_rgb[: min(256, src_rgb.shape[0])])
    rng = np.random.default_rng(cfg.seed)
    n1 = min(int(cfg.n_scalar), src_p.shape[0])
    idx1 = np.sort(rng.choice(src_p.shape[0], size=n1, replace=False))
    px = src_p[idx1, 0]
    mu, sd = float(px.mean()), float(px.std() if px.std() > 1e-8 else 1.0)
    ns = denorm_smoke(np.array([mu]), np.array([sd]))
    y_std = (px - mu) / sd
    print(f"[rtwx-o0d1] target px mean={mu:.4f} std={sd:.4f} y_std_mean={y_std.mean():.3f} y_std_std={y_std.std():.3f}", flush=True)
    tr = _train_scalar(src_rgb[idx1], y_std, epochs=int(cfg.epoch_scalar), seed=cfg.seed)
    hat_px = tr["hat"] * sd + mu
    r1_nrmse = _nrmse(hat_px, px)
    r1 = bool(r1_nrmse < R1_MAX)
    print(f"[rtwx-o0d1] R1 nrmse_px={r1_nrmse:.4f} D1_ok={r1} dtheta={tr['dtheta_from_init']:.4f}", flush=True)

    y_rand = rng.normal(size=n1).astype(np.float32)
    mu_r, sd_r = float(y_rand.mean()), float(y_rand.std() if y_rand.std() > 1e-8 else 1.0)
    tr_r = _train_scalar(src_rgb[idx1], (y_rand - mu_r) / sd_r, epochs=int(cfg.epoch_scalar), seed=cfg.seed + 1)
    hat_r = tr_r["hat"] * sd_r + mu_r
    rand_nrmse = _nrmse(hat_r, y_rand)

    r2 = False
    r2_detail: dict[str, Any] = {"skipped": True}
    if r1:
        n2 = min(int(cfg.n_pose), src_p.shape[0])
        idx2 = np.sort(rng.choice(src_p.shape[0], size=n2, replace=False))
        p2 = src_p[idx2]
        mu3, sd3 = p2.mean(0), np.where(p2.std(0) < 1e-8, 1.0, p2.std(0))
        y3 = (p2 - mu3) / sd3
        import torch
        from torch import nn

        torch.manual_seed(cfg.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(_to_nchw(src_rgb[idx2])).to(dev)
        t = torch.from_numpy(y3.astype(np.float32)).to(dev)
        model = _tiny(3, int(src_rgb.shape[1])).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
        loss_fn = nn.MSELoss()
        model.train()
        for _ep in range(int(cfg.epoch_pose)):
            opt.zero_grad()
            loss_fn(model(x), t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hat = model(x).cpu().numpy() * sd3 + mu3
        e_p = _nrmse(hat, p2)
        e_mean = _nrmse(np.repeat(p2.mean(0, keepdims=True), n2, 0), p2)
        r2 = bool(e_p < R2_MAX or e_p <= R2_MEAN_RATIO * e_mean)
        r2_detail = {"E_p": e_p, "E_p_mean": e_mean, "n": n2, "skipped": False}
        print(f"[rtwx-o0d1] R2 E_p={e_p:.4f} mean={e_mean:.4f} ok={r2}", flush=True)

    pattern = _pattern(r1, r2, r3)
    summary = {
        "header": header,
        "pattern": pattern,
        "R1": {"ok": r1, "nrmse_px": r1_nrmse, "n": n1, "hat_equals_mean": bool(np.allclose(hat_px, mu, atol=1e-3))},
        "R2": {**r2_detail, "ok": r2},
        "R3": {
            "ok": r3,
            "P_visible": vis["P_visible"],
            "frac_in_fov": vis.get("frac_in_fov"),
            "frac_occluded_in_fov": vis.get("frac_occluded_in_fov"),
            "frac_id_at_uv_equals_sim": vis.get("frac_id_at_uv_equals_sim"),
            "id_table": vis.get("id_table"),
        },
        "norm_smoke": ns,
        "target_std": {"mean": float(y_std.mean()), "std": float(y_std.std())},
        "random_label": {"nrmse": rand_nrmse, "ok": bool(rand_nrmse < R1_MAX)},
        "grad_audit": {k: tr[k] for k in ("n_param", "n_grad", "grad_norm_mean", "grad_norm_last", "dtheta_from_init", "loss0", "loss_last")},
        "image_audit": img,
        "does_not_unlock_o1": True,
        "does_not_run_o0r": True,
        "no_observability_claim": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "R1", "R2", "R3", "random_label", "norm_smoke")})
    return summary
