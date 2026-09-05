"""RTWX-O0D3: qualify head_camera + CoordConv (no GAP). No O0R. No flatten rescue."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_o0 import TASK, _cup_pose, _e_R_deg, _o0_args, _quat_fix
from .rtwx_o0d import _blob_frame, _cup_ids
from .rtwx_o0d1 import _project
from .rtwx_o0d2 import torch_linspace
from .rtwx_x0 import _nrmse
from .rtwx_x0c import _write_json
from .rtwx_x0rgb import _patch_curobo_planner, _patch_raster_shader, _resize_rgb

PREREG_PATH = "REPORT/REG/RTWX/world_perception/RTWX0O0D3_PREREG.md"
SCHEMA_ID = "aprwm.rtwx_o0d3.spatial_instrument_qualification.v1"
CAMERA = "head_camera"
SEED = 20601
RGB_SIZE = 64
N_EP = 8
N_STEPS = 32
N_SCALAR = 32
G0_FOV = 0.95
G0_VIS = 0.90
G1_MAX = 0.05
G2_MAX = 0.05
G3_MAX = 0.10
G3_MEAN_RATIO = 0.20
MED_ER = 5.0
P90_ER = 15.0
MASK_PX = 50
EPOCH_S = 600
EPOCH_P = 400
EPOCH_R = 400


@dataclass(frozen=True)
class RTWXO0D3Config:
    output: str = "runs/rtwx_o0d3"
    robotwin_repo: str = "/root/RoboTwin"
    backend: str = "robotwin"
    n_train_ep: int = N_EP
    n_steps: int = N_STEPS
    seed: int = SEED
    seed_attempts: int = 32
    max_resample: int = 16
    smoke: bool = False
    rgb_size: int = RGB_SIZE
    n_scalar: int = N_SCALAR
    epoch_s: int = EPOCH_S
    epoch_p: int = EPOCH_P
    epoch_r: int = EPOCH_R


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; RTWX-O0D3 must not write there")


def _lock(cfg: RTWXO0D3Config) -> RTWXO0D3Config:
    if cfg.backend != "robotwin" or cfg.smoke:
        return cfg
    return replace(
        cfg,
        n_train_ep=N_EP,
        n_steps=N_STEPS,
        seed=SEED,
        rgb_size=RGB_SIZE,
        n_scalar=N_SCALAR,
        epoch_s=EPOCH_S,
        epoch_p=EPOCH_P,
        epoch_r=EPOCH_R,
    )


def _head_cam(env: Any) -> Any:
    for cam, name in zip(env.cameras.static_camera_list, env.cameras.static_camera_name):
        if name == CAMERA:
            return cam
    raise RuntimeError("head_camera missing")


def _capture_head(env: Any, size: int) -> np.ndarray:
    env._update_render()
    env.cameras.update_picture()
    rgb_map = env.cameras.get_rgb()
    cam = rgb_map.get(CAMERA)
    if cam is None or cam.get("rgb") is None:
        raise RuntimeError(f"missing {CAMERA}; keys={list(rgb_map)}")
    return _resize_rgb(cam["rgb"], size)


def _coord_net(n_out: int, size: int):
    from torch import nn

    spat = 32 * (size // 4) * (size // 4)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(5, 16, 5, stride=2, padding=2)
            self.c2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(spat, 128), nn.ReLU(), nn.Linear(128, n_out))

        def forward(self, x):
            import torch

            b, _, h, w = x.shape
            yy = torch_linspace(h, x.device).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch_linspace(w, x.device).view(1, 1, 1, w).expand(b, 1, h, w)
            z = torch.cat([x, xx, yy], dim=1)
            return self.fc(torch.relu(self.c2(torch.relu(self.c1(z)))))

    return M()


def _nchw(rgb: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(rgb, dtype=np.float32) / 255.0, (0, 3, 1, 2))


def _fit_torch_nd(model: Any, x: Any, y: np.ndarray, *, epochs: int, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
    else:
        x = x.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x), t).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(x).detach().cpu().numpy()


def _predict(rgb: np.ndarray, y: np.ndarray, *, epochs: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    mu, sd = y.mean(0), np.where(y.std(0) < 1e-8, 1.0, y.std(0))
    ystd = (y - mu) / sd
    hat_z = _fit_torch_nd(_coord_net(y.shape[1], int(rgb.shape[1])), _nchw(rgb), ystd, epochs=epochs, seed=seed)
    return hat_z.reshape(y.shape) * sd + mu


def _quat_R(q: np.ndarray) -> np.ndarray:
    q = _quat_fix(np.asarray(q, dtype=np.float64).reshape(-1, 4))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _R_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    out = np.zeros((R.shape[0], 4))
    for i, m in enumerate(R):
        tr = float(np.trace(m))
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            out[i] = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        else:
            out[i] = [1.0, 0, 0, 0]
    return _quat_fix(out)


def _rot_axis(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-8)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _e_R_after_yaw(qhat: np.ndarray, qgt: np.ndarray) -> np.ndarray:
    Rh, Rg = _quat_R(qhat), _quat_R(qgt)
    best = np.full(qhat.shape[0], 180.0)
    thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for th in thetas:
        aligned = []
        for i in range(qgt.shape[0]):
            axis = Rg[i] @ np.array([0.0, 0.0, 1.0])
            Rth = _rot_axis(axis, th) @ Rh[i]
            aligned.append(Rth)
        qn = _R_quat(np.stack(aligned))
        best = np.minimum(best, _e_R_deg(qn, qgt))
    return best


def collect_head(cfg: RTWXO0D3Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.backend == "numpy":
        n = int(cfg.n_train_ep) * int(cfg.n_steps)
        ps, qs, rgbs, vis, fov = [], [], [], [], []
        for i in range(n):
            p = np.array([-0.28 + 0.56 * (i % 32) / 31.0, -0.1 + 0.01 * (i // 32), 0.71])
            q = _quat_fix(np.array([1.0, 0.05 * np.sin(i / 7.0), 0.0, 0.0]))[0]
            rgb, mask = _blob_frame(p, cfg.rgb_size)
            rgb = rgb.copy()
            rgb[0, i % 64, 2] = 40 + (i % 200)
            ps.append(p)
            qs.append(q)
            rgbs.append(rgb)
            vis.append(True)
            fov.append(True)
        return {
            "p": np.stack(ps),
            "quat": np.stack(qs),
            "rgb": np.stack(rgbs),
            "in_fov": np.asarray(fov),
            "visible": np.asarray(vis),
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
    n_ep, n_steps = int(cfg.n_train_ep), int(cfg.n_steps)
    recs: list[dict[str, Any]] = []
    attempts, slot = 0, 0
    while len(recs) < n_ep and attempts < n_ep * max(2, cfg.max_resample):
        attempts += 1
        seed = int(cfg.seed + slot * 17 + attempts)
        slot += 1
        print(f"[rtwx-o0d3] collect {len(recs)}/{n_ep} attempt {attempts}", flush=True)
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
            tt = np.arange(n_steps) * (1.0 / 250.0 * 50.0)
            qtar_l, _ = _multisine(q0_l, lo_l, hi_l, tt, rng)
            qtar_r, _ = _multisine(q0_r, lo_r, hi_r, tt, rng)
            cam = _head_cam(env)
            frames = []
            bad = None
            for i in range(n_steps):
                p, quat = _cup_pose(env)
                rgb = _capture_head(env, cfg.rgb_size)
                try:
                    seg = np.asarray(cam.get_picture("Segmentation"))
                    actor = np.asarray(seg[..., 1]).astype(np.int32)
                except Exception:
                    actor = np.zeros((480, 640), dtype=np.int32)
                w, h = actor.shape[1], actor.shape[0]
                K = np.asarray(cam.get_intrinsic_matrix())
                E = np.asarray(cam.get_extrinsic_matrix())
                pr = _project(K, E, p, w, h)
                uu = int(np.clip(round(pr["u"]), 0, w - 1))
                vv = int(np.clip(round(pr["v"]), 0, h - 1))
                id_uv = int(actor[vv, uu]) if pr["in_fov"] else -1
                frames.append({"p": p, "quat": quat, "rgb": rgb, "actor": actor, "pr": pr, "id_uv": id_uv, "sim_ids": sorted(sim_ids)})
                action = _build_action(env, qtar_l[i], qtar_r[i])
                orig, box = _count_steps(env)
                cnt0 = int(getattr(env, "take_action_cnt", 0))
                env.take_action(action, action_type="qpos")
                env.scene.step = orig
                if int(getattr(env, "take_action_cnt", 0)) == cnt0 or _contact_invalid(env):
                    bad = "step"
                    break
            env.close()
            if bad or len(frames) != n_steps:
                continue
            recs.append(frames)
        except Exception as exc:
            print(f"[rtwx-o0d3] fail {type(exc).__name__}: {exc}", flush=True)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    if len(recs) < n_ep:
        raise RuntimeError(f"O0D3 collect {len(recs)}/{n_ep}")
    ps, qs, rgbs, fov, vis = [], [], [], [], []
    for frames in recs:
        repaired = None
        for fr in frames:
            if fr["pr"]["in_fov"] and fr["id_uv"] >= 0:
                repaired = int(fr["id_uv"])
                break
        for fr in frames:
            actor = fr["actor"]
            cid = repaired if repaired is not None else (fr["sim_ids"][0] if fr["sim_ids"] else -1)
            area = int((actor == cid).sum()) if cid >= 0 else 0
            in_fov = bool(fr["pr"]["in_fov"])
            visible = bool(in_fov and (area >= MASK_PX or fr["id_uv"] >= 0))
            ps.append(fr["p"])
            qs.append(fr["quat"])
            rgbs.append(fr["rgb"])
            fov.append(in_fov)
            vis.append(visible)
    return {
        "p": np.stack(ps),
        "quat": np.stack(qs),
        "rgb": np.stack(rgbs),
        "in_fov": np.asarray(fov),
        "visible": np.asarray(vis),
    }


def _pattern(g0: bool, g1: bool, g2: bool, g3: bool, g4: bool, sym: bool) -> str:
    if not g0:
        return "coverage_failure"
    if not g1:
        return "scalar_spatial_failure"
    if not g2:
        return "capacity_control_failure"
    if not g3:
        return "position_memorization_failure"
    if not g4 and not sym:
        return "rotation_memorization_failure"
    if not g4 and sym:
        return "rotation_symmetry_limited"
    return "instrument_qualified"


def run_rtwx_o0d3(output: str | Path, config: RTWXO0D3Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or RTWXO0D3Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    header = {
        "stage": "RTWX-O0D3",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "camera": CAMERA,
        "no_gap": True,
        "no_flatten_gate": True,
        "does_not_run_o0r": True,
        "does_not_unlock_o1": True,
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    _write_json(root / "header.json", header)
    pool = collect_head(cfg)
    p_fov = float(np.mean(pool["in_fov"]))
    p_vis = float(np.mean(pool["visible"]))
    g0 = bool(p_fov >= G0_FOV and p_vis >= G0_VIS)
    print(f"[rtwx-o0d3] G0 FOV={p_fov:.3f} vis={p_vis:.3f} ok={g0}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    n = int(pool["p"].shape[0])
    n1 = min(int(cfg.n_scalar), n)
    i1 = np.sort(rng.choice(n, size=n1, replace=False))
    px = pool["p"][i1, 0]
    rgb1 = pool["rgb"][i1]
    hat_px = _predict(rgb1, px, epochs=int(cfg.epoch_s), seed=cfg.seed).reshape(-1)
    g1_n = _nrmse(hat_px, px)
    g1 = bool(g1_n < G1_MAX)
    print(f"[rtwx-o0d3] G1 nrmse_px={g1_n:.4f} ok={g1}", flush=True)

    y_rand = (px.mean() + (px.std() if px.std() > 1e-8 else 1.0) * rng.normal(size=n1))
    hat_r = _predict(rgb1, y_rand, epochs=int(cfg.epoch_s), seed=cfg.seed + 1).reshape(-1)
    g2_n = _nrmse(hat_r, y_rand)
    g2 = bool(g2_n < G2_MAX)
    print(f"[rtwx-o0d3] G2 nrmse_rand={g2_n:.4f} ok={g2}", flush=True)

    hat_p = _predict(pool["rgb"], pool["p"], epochs=int(cfg.epoch_p), seed=cfg.seed + 2)
    e_p = _nrmse(hat_p, pool["p"])
    e_mean = _nrmse(np.repeat(pool["p"].mean(0, keepdims=True), n, 0), pool["p"])
    g3 = bool(e_p < G3_MAX or e_p <= G3_MEAN_RATIO * e_mean)
    print(f"[rtwx-o0d3] G3 E_p={e_p:.4f} mean={e_mean:.4f} ok={g3}", flush=True)

    q = pool["quat"]
    hat_q = _quat_fix(_predict(pool["rgb"], q, epochs=int(cfg.epoch_r), seed=cfg.seed + 3))
    er = _e_R_deg(hat_q, q)
    med, p90 = float(np.median(er)), float(np.percentile(er, 90))
    g4 = bool(med < MED_ER and p90 < P90_ER)
    er_yaw = _e_R_after_yaw(hat_q, q)
    med_y, p90_y = float(np.median(er_yaw)), float(np.percentile(er_yaw, 90))
    sym = bool((not g4) and med_y < MED_ER and p90_y < P90_ER)
    print(f"[rtwx-o0d3] G4 med={med:.2f} p90={p90:.2f} yaw_med={med_y:.2f} sym={sym} ok={g4}", flush=True)

    pattern = _pattern(g0, g1, g2, g3, g4, sym)
    summary = {
        "header": header,
        "pattern": pattern,
        "G0": {"ok": g0, "P_FOV": p_fov, "P_visible": p_vis, "camera": CAMERA},
        "G1": {"ok": g1, "nrmse_px": g1_n, "n": n1},
        "G2": {"ok": g2, "nrmse_rand": g2_n},
        "G3": {"ok": g3, "E_p": e_p, "E_p_mean": e_mean, "n": n},
        "G4": {"ok": g4, "median_e_R": med, "p90_e_R": p90, "median_after_yaw": med_y, "p90_after_yaw": p90_y, "symmetry_limited": sym},
        "unlocks_o0r": pattern == "instrument_qualified",
        "does_not_run_o0r": True,
        "does_not_unlock_o1": True,
        "no_observability_claim": True,
    }
    _write_json(root / "run.json", summary)
    _write_json(root / "summary.json", summary)
    _write_json(root / "metrics.json", {k: summary[k] for k in ("pattern", "G0", "G1", "G2", "G3", "G4")})
    return summary
