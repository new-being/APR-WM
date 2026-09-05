"""symx_rigid.v1: oracle rigid body + table; no RGB in the state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HOST_ID = "symx_rigid.v1"
GRAVITY = np.array([0.0, 0.0, -9.81])
L_REF, V_REF, W_REF = 0.05, 0.30, 5.0


def _hat(w: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _exp_SO3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    K = _hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _geodesic_rad(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(c))


def axis_angle_R(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    return _exp_SO3(a * np.radians(float(deg)))


def d_state(s1: dict[str, np.ndarray], s2: dict[str, np.ndarray]) -> float:
    return float(
        np.linalg.norm(s1["p"] - s2["p"]) / L_REF
        + _geodesic_rad(s1["R"], s2["R"]) / np.pi
        + np.linalg.norm(s1["v"] - s2["v"]) / V_REF
        + np.linalg.norm(s1["w"] - s2["w"]) / W_REF
    )


def apply_g(s: dict[str, np.ndarray], g: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray(g, dtype=np.float64)
    return {"p": s["p"].copy(), "R": s["R"] @ g, "v": s["v"].copy(), "w": g.T @ s["w"]}


def apply_g_inv(s: dict[str, np.ndarray], g: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray(g, dtype=np.float64)
    return {"p": s["p"].copy(), "R": s["R"] @ g.T, "v": s["v"].copy(), "w": g @ s["w"]}


def transform_action(action: dict[str, Any], g: np.ndarray) -> dict[str, Any]:
    out = dict(action)
    g = np.asarray(g, dtype=np.float64)
    kind = action["kind"]
    if kind in ("body_point", "body_torque"):
        if action.get("x_O") is not None:
            out["x_O"] = g.T @ np.asarray(action["x_O"], dtype=np.float64)
        if action.get("j_B") is not None:
            out["j_B"] = g.T @ np.asarray(action["j_B"], dtype=np.float64)
        if action.get("tau_B") is not None:
            out["tau_B"] = g.T @ np.asarray(action["tau_B"], dtype=np.float64)
    return out


@dataclass(frozen=True)
class Body:
    name: str
    mass: float
    I_B: np.ndarray  # (3,) principal, object frame
    x_sph: np.ndarray  # (n,3) object-frame contact spheres
    r_sph: np.ndarray
    visual_logo: np.ndarray | None = None  # unused by D_H
    mu: float = 0.35
    mu_aniso: tuple[float, float] = (0.0, 0.0)  # (a_x, a_z) yaw-dependent table friction


def _rings(radius: float, height: float, n_around: int = 24, n_h: int = 5) -> np.ndarray:
    ys = np.linspace(-0.5 * height, 0.5 * height, n_h)
    ang = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    pts = []
    for y in ys:
        for t in ang:
            pts.append([radius * np.cos(t), y, radius * np.sin(t)])
    return np.asarray(pts, dtype=np.float64)


def make_bodies() -> dict[str, Body]:
    m = 0.20
    r, h = 0.035, 0.08
    Iy = 0.5 * m * r * r
    Ixz = 0.25 * m * r * r + (1.0 / 12.0) * m * h * h
    cyl = _rings(r, h)
    r_c = np.full(cyl.shape[0], 0.008)
    c0 = Body("C0", m, np.array([Ixz, Iy, Ixz]), cyl, r_c)
    handle = np.array([[0.055, 0.02, 0.015]], dtype=np.float64)
    x1 = np.concatenate([cyl, handle], 0)
    # parallel-axis blob
    m_h = 0.06
    I1 = c0.I_B + m_h * np.array([handle[0, 1] ** 2 + handle[0, 2] ** 2, handle[0, 0] ** 2 + handle[0, 2] ** 2, handle[0, 0] ** 2 + handle[0, 1] ** 2])
    c1 = Body("C1", m + m_h, I1, x1, np.concatenate([r_c, [0.012]]))
    # box a≠b≠c
    lx, ly, lz = 0.08, 0.045, 0.03
    gx, gy, gz = np.meshgrid(np.linspace(-lx / 2, lx / 2, 3), np.linspace(-ly / 2, ly / 2, 3), np.linspace(-lz / 2, lz / 2, 3), indexing="ij")
    xbox = np.stack([gx, gy, gz], -1).reshape(-1, 3)
    Ibox = (m / 12.0) * np.array([ly * ly + lz * lz, lx * lx + lz * lz, lx * lx + ly * ly])
    c2 = Body("C2", m, Ibox, xbox, np.full(xbox.shape[0], 0.008))
    c3 = Body("C3", m, np.full(3, 0.4 * m * 0.04**2), np.zeros((1, 3)), np.array([0.04]))
    c4 = Body("C4", m, np.array([0.45 * Ixz, Iy, 2.2 * Ixz]), cyl.copy(), r_c.copy())
    c5 = Body("C5", m, c0.I_B.copy(), cyl.copy(), r_c.copy(), visual_logo=np.array([1.0, 0.0, 0.0]))
    c_fr = Body("C0fr", m, c0.I_B.copy(), cyl.copy(), r_c.copy(), mu_aniso=(4.0, 0.0))
    return {"C0": c0, "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C0fr": c_fr}


def _contact(body: Body, s: dict[str, np.ndarray], k: float = 500.0, c: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    R, p, v, w = s["R"], s["p"], s["v"], s["w"]
    mu = float(body.mu)
    Xw = p.reshape(1, 3) + body.x_sph @ R.T
    wW = R @ w
    ri = Xw - p.reshape(1, 3)
    vi = v.reshape(1, 3) + np.cross(np.broadcast_to(wW, ri.shape), ri)
    z = Xw[:, 2] - body.r_sph
    hit = z < 0.0
    if not np.any(hit):
        return np.zeros(3), np.zeros(3)
    vn = vi[hit, 2]
    Fn_mag = np.clip(-k * z[hit] - c * np.minimum(vn, 0.0), 0.0, 80.0)
    Fn = np.stack([np.zeros(hit.sum()), np.zeros(hit.sum()), Fn_mag], 1)
    vt = vi[hit].copy()
    vt[:, 2] = 0.0
    speed = np.linalg.norm(vt, axis=1, keepdims=True)
    vt_hat = vt / (speed + 1e-4)
    ax, az = (float(body.mu_aniso[0]), float(body.mu_aniso[1]))
    rx = R[:, 0].copy()
    rz = R[:, 2].copy()
    rx[2] = 0.0
    rz[2] = 0.0
    rx = rx / max(float(np.linalg.norm(rx)), 1e-12)
    rz = rz / max(float(np.linalg.norm(rz)), 1e-12)
    cx = vt_hat @ rx
    cz = vt_hat @ rz
    mu_eff = mu * (1.0 + ax * cx * cx + az * cz * cz)
    Ft = -mu_eff.reshape(-1, 1) * np.abs(Fn_mag).reshape(-1, 1) * vt / (speed + 1e-4)
    Fi = Fn + Ft
    return Fi.sum(0), np.cross(ri[hit], Fi).sum(0)


def step(body: Body, s: dict[str, np.ndarray], action: dict[str, Any] | None, dt: float, *, apply_impulse: bool) -> dict[str, np.ndarray]:
    R, p, v, w = s["R"].copy(), s["p"].copy(), s["v"].copy(), s["w"].copy()
    F = body.mass * GRAVITY
    tauW = np.zeros(3)
    Fc, tc = _contact(body, s)
    F = F + Fc
    tauW = tauW + tc
    if apply_impulse and action is not None:
        kind = action["kind"]
        if kind == "cm_world":
            F = F + np.asarray(action["j_W"], dtype=np.float64) / dt
        elif kind == "body_point":
            j_B = np.asarray(action["j_B"], dtype=np.float64)
            x_O = np.asarray(action["x_O"], dtype=np.float64)
            j_W = R @ j_B
            F = F + j_W / dt
            tauW = tauW + np.cross(R @ x_O, j_W) / dt
        elif kind == "body_torque":
            tauW = tauW + (R @ np.asarray(action["tau_B"], dtype=np.float64)) / dt
    a = F / body.mass
    tauB = R.T @ tauW
    I = np.asarray(body.I_B, dtype=np.float64)
    Iw = I * w
    wdot = (tauB - np.cross(w, Iw)) / np.maximum(I, 1e-9)
    # World-frame |v| and body-frame |ω| isotropic caps — axis-aligned
    # clip(w_i) breaks object-frame SO(2) (cube invariant only under 90°).
    v = v + dt * a
    vn = float(np.linalg.norm(v))
    if vn > 8.0:
        v = v * (8.0 / vn)
    w = w + dt * wdot
    wn = float(np.linalg.norm(w))
    if wn > 40.0:
        w = w * (40.0 / wn)
    p = p + dt * v
    if not np.isfinite(p).all() or not np.isfinite(w).all():
        return {"p": np.zeros(3), "R": np.eye(3), "v": np.zeros(3), "w": np.zeros(3)}
    R = R @ _exp_SO3(dt * w)
    return {"p": p, "R": R, "v": v, "w": w}


def rollout(body: Body, s0: dict[str, np.ndarray], action: dict[str, Any], H: int, dt: float) -> list[dict[str, np.ndarray]]:
    s = {k: np.asarray(v, dtype=np.float64).copy() for k, v in s0.items()}
    traj = [s]
    for t in range(H):
        s = step(body, s, action, dt, apply_impulse=(t == 0))
        traj.append(s)
    return traj
