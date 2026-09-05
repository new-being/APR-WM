"""SYM-X1: quotient representation utility. No RGB. Ĝ frozen from X0."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .rtwx_x0c import _write_json
from .symx_plant import (
    HOST_ID,
    L_REF,
    V_REF,
    W_REF,
    _exp_SO3,
    apply_g,
    make_bodies,
    rollout,
    transform_action,
)
from .symx_x0 import (
    CONDITIONS,
    _candidates,
    in_Gstar,
    sample_action,
    sample_state,
)

PREREG_PATH = "REPORT/REG/SYMX/SYMX1_PREREG.md"
SCHEMA_ID = "aprwm.symx_x1.quotient_utility.v1"
SEEDS = (41101, 41102, 41103)
X0_DEFAULT = "runs/symx_x0"
DT = 0.005
H = 40
N_PROBE = 256
WIDTHS = (4, 8, 16, 32, 64)
W_MATCH = 32
Z_DIM = 16
EPOCHS = 60
EPS_NI = 1.05
G0_RATIO = 0.80
YAW_R2_MAX = 0.10
AXIS_R2_MIN = 0.80
KINDS = ("none", "cm_world", "body_point", "body_torque")
EY = np.array([0.0, 1.0, 0.0])
EX = np.array([1.0, 0.0, 0.0])
FEW_K = (32, 128, 256)


@dataclass(frozen=True)
class SYMX1Config:
    output: str = "runs/symx_x1"
    x0_summary: str = X0_DEFAULT
    seeds: tuple[int, ...] = SEEDS
    n_probe: int = N_PROBE
    horizon: int = H
    dt: float = DT
    widths: tuple[int, ...] = WIDTHS
    w_match: int = W_MATCH
    epochs: int = EPOCHS
    smoke: bool = False


def _lock(cfg: SYMX1Config) -> SYMX1Config:
    if cfg.smoke:
        return replace(cfg, n_probe=16, horizon=16, seeds=(51, 52, 53), widths=(8, 32), w_match=32, epochs=25)
    return replace(cfg, seeds=SEEDS, n_probe=N_PROBE, horizon=H, dt=DT, widths=WIDTHS, w_match=W_MATCH, epochs=EPOCHS)


def _refuse_r10(output: Path) -> None:
    if "r10_c0" in str(output.resolve()):
        raise RuntimeError("r10_c0 is locked; SYM-X1 must not write there")


def _require_x0(path: Path) -> dict[str, Any]:
    p = path / "summary.json" if path.is_dir() else path
    if not p.is_file():
        raise RuntimeError(f"SYM-X1 requires X0 summary at {p}")
    s = json.loads(p.read_text(encoding="utf-8"))
    if s.get("pattern") != "causal_symmetry_supported":
        raise RuntimeError(f"SYM-X1 locked until X0 causal_symmetry_supported; got {s.get('pattern')}")
    return s


def _cand_map() -> dict[str, np.ndarray]:
    return {n: g for n, g in _candidates()}


def _g_star(cond: str) -> np.ndarray:
    mats = [g for _, g in _candidates() if in_Gstar(g, cond)]
    return np.stack(mats, 0)


def _g_hat(x0: dict[str, Any], cond: str, cmap: dict[str, np.ndarray]) -> np.ndarray:
    names = x0["per_condition"][cond]["accepted"]
    return np.stack([cmap[n] for n in names], 0)


def _geo_batch(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R, axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(c)


def _d_np(p1, R1, v1, w1, p2, R2, v2, w2) -> np.ndarray:
    geo = _geo_batch(np.swapaxes(R1, -1, -2) @ R2)
    return np.linalg.norm(p1 - p2, axis=-1) / L_REF + geo / np.pi + np.linalg.norm(v1 - v2, axis=-1) / V_REF + np.linalg.norm(w1 - w2, axis=-1) / W_REF


def d_G_np(p1, R1, v1, w1, p2, R2, v2, w2, G: np.ndarray) -> np.ndarray:
    """min_g d(s1, g·s2). G: (K,3,3). Broadcasts over leading batch dims of s1/s2."""
    R2g = np.einsum("...ij,kjl->...kil", R2, G)
    w2g = np.einsum("kji,...j->...ki", G, w2)
    p2e = np.expand_dims(p2, -2)
    v2e = np.expand_dims(v2, -2)
    p1e = np.expand_dims(p1, -2)
    v1e = np.expand_dims(v1, -2)
    R1e = np.expand_dims(R1, -3)
    w1e = np.expand_dims(w1, -2)
    return _d_np(p1e, R1e, v1e, w1e, p2e, R2g, v2e, w2g).min(-1)


def g_star_of(R: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Discrete section: lex-least {R g : g in G} (G-invariant representative)."""
    R = np.asarray(R, dtype=np.float64)
    Rg = np.einsum("ij,kjl->kil", R, G)
    flat = Rg.reshape(G.shape[0], 9)
    k = int(np.lexsort(tuple(flat[:, i] for i in range(8, -1, -1)))[0])
    return G[k]


def canon_state(s: dict[str, np.ndarray], G: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    g = g_star_of(s["R"], G)
    return apply_g(s, g), g


def pack_action(a: dict[str, Any]) -> np.ndarray:
    k = np.zeros(4, dtype=np.float64)
    k[KINDS.index(a["kind"])] = 1.0
    jw = np.asarray(a.get("j_W") if a.get("j_W") is not None else np.zeros(3), dtype=np.float64)
    jb = np.asarray(a.get("j_B") if a.get("j_B") is not None else np.zeros(3), dtype=np.float64)
    xo = np.asarray(a.get("x_O") if a.get("x_O") is not None else np.zeros(3), dtype=np.float64)
    tb = np.asarray(a.get("tau_B") if a.get("tau_B") is not None else np.zeros(3), dtype=np.float64)
    return np.concatenate([k, jw, jb, xo, tb])


def pack_state(s: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([s["p"], s["R"].reshape(9), s["v"], s["w"]])


def unpack_state(x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"p": x[0:3].copy(), "R": x[3:12].reshape(3, 3).copy(), "v": x[12:15].copy(), "w": x[15:18].copy()}


def n_axis(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float64) @ EY


def yaw_cs(R: np.ndarray) -> np.ndarray:
    n = n_axis(R)
    nn = float(np.linalg.norm(n))
    n = n / max(nn, 1e-12)
    x = np.asarray(R, dtype=np.float64) @ EX
    x = x - n * float(np.dot(x, n))
    xn = float(np.linalg.norm(x))
    x = x / max(xn, 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(np.dot(n, EX))) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, ref)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(n, e1)
    return np.array([float(np.dot(x, e1)), float(np.dot(x, e2))])


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    v = float(np.mean((y - y.mean(0)) ** 2))
    if v < 1e-12:
        return 1.0 if float(np.mean((y - yhat) ** 2)) < 1e-12 else 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / v)


def ridge_r2(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, l2: float = 1e-2) -> float:
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_te = np.asarray(x_te, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    y_te = np.asarray(y_te, dtype=np.float64)
    if y_tr.ndim == 1:
        y_tr = y_tr[:, None]
        y_te = y_te.reshape(-1, 1)
    xb = np.concatenate([x_tr, np.ones((x_tr.shape[0], 1))], 1)
    xe = np.concatenate([x_te, np.ones((x_te.shape[0], 1))], 1)
    a = xb.T @ xb + l2 * np.eye(xb.shape[1])
    w = np.linalg.solve(a, xb.T @ y_tr)
    return _r2(y_te, xe @ w)


def axis_r2(x_tr, n_tr, x_te, n_te) -> float:
    r = ridge_r2(x_tr, n_tr, x_te, n_te)
    r_m = ridge_r2(x_tr, n_tr, x_te, -n_te)
    return float(max(r, r_m))


def mlp_params(in_dim: int, hidden: int, out_dim: int) -> int:
    return in_dim * hidden + hidden + hidden * hidden + hidden + hidden * out_dim + out_dim


def latent_params(z: int, hidden: int) -> int:
    enc = mlp_params(18, hidden, z)
    dyn = mlp_params(z + 16, hidden, z)
    dec = mlp_params(z, hidden, 12)
    return enc + dyn + dec


def _collect(body, n_probe: int, seed: int, horizon: int, dt: float):
    rng = np.random.default_rng(int(seed))
    trajs, acts = [], []
    none = {"kind": "none"}
    for i in range(n_probe):
        s0 = sample_state(rng, airborne=(i % 3 == 0))
        a = sample_action(rng)
        trajs.append(rollout(body, s0, a, horizon, dt))
        acts.append(a)
    return trajs, acts, none


def _transitions(trajs, acts, none, G_in: np.ndarray | None):
    p, R, v, w, af, p2, R2, v2, w2 = [], [], [], [], [], [], [], [], []
    feat, yaw, ax = [], [], []
    for tr, a0 in zip(trajs, acts):
        for t in range(len(tr) - 1):
            s = {k: tr[t][k].copy() for k in ("p", "R", "v", "w")}
            sp = {k: tr[t + 1][k].copy() for k in ("p", "R", "v", "w")}
            a = a0 if t == 0 else none
            if G_in is not None:
                s, g = canon_state(s, G_in)
                a = transform_action(a, g)
                sp = apply_g(sp, g)
            p.append(s["p"])
            R.append(s["R"])
            v.append(s["v"])
            w.append(s["w"])
            af.append(pack_action(a))
            p2.append(sp["p"])
            R2.append(sp["R"])
            v2.append(sp["v"])
            w2.append(sp["w"])
            feat.append(pack_state(s))
            yaw.append(yaw_cs(tr[t]["R"]))
            ax.append(n_axis(tr[t]["R"]))
    return {
        "p": np.stack(p),
        "R": np.stack(R),
        "v": np.stack(v),
        "w": np.stack(w),
        "a": np.stack(af),
        "p2": np.stack(p2),
        "R2": np.stack(R2),
        "v2": np.stack(v2),
        "w2": np.stack(w2),
        "feat": np.stack(feat),
        "yaw": np.stack(yaw),
        "ax": np.stack(ax),
        "n_ep": len(trajs),
        "H": len(trajs[0]) - 1,
        "trajs": trajs,
        "acts": acts,
        "none": none,
    }


def _apply_res(p, R, v, w, res: np.ndarray):
    res = np.asarray(res, dtype=np.float64)
    dp = np.clip(res[..., 0:3], -0.25, 0.25)
    dth = np.clip(res[..., 3:6], -1.5, 1.5)
    dv = np.clip(res[..., 6:9], -4.0, 4.0)
    dw = np.clip(res[..., 9:12], -15.0, 15.0)
    n = int(np.prod(p.shape[:-1]))
    Rn = np.empty((n, 3, 3), dtype=np.float64)
    RR = np.asarray(R, dtype=np.float64).reshape(n, 3, 3)
    dtn = dth.reshape(n, 3)
    for i in range(n):
        Rn[i] = RR[i] @ _exp_SO3(dtn[i])
    return p + dp, Rn.reshape(R.shape), v + dv, w + dw


def _apply_res_t(p, R, v, w, res):
    import torch

    dp = res[:, 0:3].clamp(-0.25, 0.25)
    dth = res[:, 3:6].clamp(-1.5, 1.5)
    dv = res[:, 6:9].clamp(-4.0, 4.0)
    dw = res[:, 9:12].clamp(-15.0, 15.0)
    return p + dp, torch.matmul(R, _exp_t(dth)), v + dv, w + dw


def _dG_t(p, R, v, w, p2, R2, v2, w2, Gt):
    import torch

    R2g = torch.einsum("bij,njk->bnik", R2, Gt)
    w2g = torch.einsum("nji,bj->bnj", Gt, w2)
    ch = _chordal_t(R.unsqueeze(1).expand_as(R2g), R2g)
    d = (
        (p.unsqueeze(1) - p2.unsqueeze(1)).norm(dim=-1) / L_REF
        + ch
        + (v.unsqueeze(1) - v2.unsqueeze(1)).norm(dim=-1) / V_REF
        + (w.unsqueeze(1) - w2g).norm(dim=-1) / W_REF
    )
    return d.min(1).values.mean()


def _train_mlp(data: dict[str, np.ndarray], *, hidden: int, G_loss: np.ndarray, epochs: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    n_ep, H = int(data["n_ep"]), int(data["H"])
    feat = data["feat"].reshape(n_ep, H, -1)
    aa = data["a"].reshape(n_ep, H, -1)
    pA = data["p"].reshape(n_ep, H, 3)
    RA = data["R"].reshape(n_ep, H, 3, 3)
    vA = data["v"].reshape(n_ep, H, 3)
    wA = data["w"].reshape(n_ep, H, 3)
    p2 = data["p2"].reshape(n_ep, H, 3)
    R2 = data["R2"].reshape(n_ep, H, 3, 3)
    v2 = data["v2"].reshape(n_ep, H, 3)
    w2 = data["w2"].reshape(n_ep, H, 3)
    x = np.concatenate([data["feat"], data["a"]], 1).astype(np.float32)
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-6)
    xt = torch.from_numpy(((x - mu) / sd).astype(np.float32))
    p1t = torch.from_numpy(data["p"].astype(np.float32))
    R1t = torch.from_numpy(data["R"].astype(np.float32))
    v1t = torch.from_numpy(data["v"].astype(np.float32))
    w1t = torch.from_numpy(data["w"].astype(np.float32))
    p2t = torch.from_numpy(data["p2"].astype(np.float32))
    R2t = torch.from_numpy(data["R2"].astype(np.float32))
    v2t = torch.from_numpy(data["v2"].astype(np.float32))
    w2t = torch.from_numpy(data["w2"].astype(np.float32))
    Gt = torch.from_numpy(G_loss.astype(np.float32))
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 12))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    mu_t = torch.from_numpy(mu.astype(np.float32))
    sd_t = torch.from_numpy(sd.astype(np.float32))
    K = 4 if H >= 4 else max(H, 1)
    rng = np.random.default_rng(int(seed) + 3)

    def _norm_ba(feat_b, a_b):
        return (torch.cat([feat_b, a_b], -1) - mu_t) / sd_t

    for _ in range(epochs):
        res = net(xt)
        pn, Rn, vn, wn = _apply_res_t(p1t, R1t, v1t, w1t, res)
        loss_1 = _dG_t(pn, Rn, vn, wn, p2t, R2t, v2t, w2t, Gt)
        n_b = min(n_ep, 64)
        ie = rng.choice(n_ep, size=n_b, replace=False)
        t0 = rng.integers(0, H - K + 1, size=n_b)
        p = torch.from_numpy(pA[ie, t0].astype(np.float32))
        R = torch.from_numpy(RA[ie, t0].astype(np.float32))
        v = torch.from_numpy(vA[ie, t0].astype(np.float32))
        w = torch.from_numpy(wA[ie, t0].astype(np.float32))
        loss_k = 0.0
        for k in range(K):
            ab = torch.from_numpy(aa[ie, t0 + k].astype(np.float32))
            f = torch.cat([p, R.reshape(n_b, 9), v, w], -1) if k > 0 else torch.from_numpy(feat[ie, t0 + k].astype(np.float32))
            p, R, v, w = _apply_res_t(p, R, v, w, net(_norm_ba(f, ab)))
            loss_k = loss_k + _dG_t(
                p,
                R,
                v,
                w,
                torch.from_numpy(p2[ie, t0 + k].astype(np.float32)),
                torch.from_numpy(R2[ie, t0 + k].astype(np.float32)),
                torch.from_numpy(v2[ie, t0 + k].astype(np.float32)),
                torch.from_numpy(w2[ie, t0 + k].astype(np.float32)),
                Gt,
            )
        loss = loss_1 + loss_k / K
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

    def hidden_of(feat, a):
        inp = np.concatenate([feat, a], 1)
        z = np.clip((inp - mu) / sd, -30, 30).astype(np.float32)
        with torch.no_grad():
            h = net[2](net[1](net[0](torch.from_numpy(z))))
        return h.numpy().astype(np.float64)

    def predict_res(feat, a):
        inp = np.concatenate([np.asarray(feat, dtype=np.float64), np.asarray(a, dtype=np.float64)], 1)
        z = np.clip((inp - mu) / sd, -30, 30).astype(np.float32)
        with torch.no_grad():
            return net(torch.from_numpy(z)).numpy().astype(np.float64)

    n_param = int(sum(pp.numel() for pp in net.parameters()))
    return predict_res, hidden_of, n_param


def _train_latent(data: dict[str, np.ndarray], *, hidden: int, z_dim: int, epochs: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    s = data["feat"].astype(np.float32)
    a = data["a"].astype(np.float32)
    mu, sd = s.mean(0), np.maximum(s.std(0), 1e-6)
    mua, sda = a.mean(0), np.maximum(a.std(0), 1e-6)
    st = torch.from_numpy(((s - mu) / sd).astype(np.float32))
    at = torch.from_numpy(((a - mua) / sda).astype(np.float32))
    p = torch.from_numpy(data["p"].astype(np.float32))
    R = torch.from_numpy(data["R"].astype(np.float32))
    v = torch.from_numpy(data["v"].astype(np.float32))
    w = torch.from_numpy(data["w"].astype(np.float32))
    p2 = torch.from_numpy(data["p2"].astype(np.float32))
    R2 = torch.from_numpy(data["R2"].astype(np.float32))
    v2 = torch.from_numpy(data["v2"].astype(np.float32))
    w2 = torch.from_numpy(data["w2"].astype(np.float32))
    enc = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(), nn.Linear(hidden, z_dim))
    dyn = nn.Sequential(nn.Linear(z_dim + 16, hidden), nn.SiLU(), nn.Linear(hidden, z_dim))
    dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.SiLU(), nn.Linear(hidden, 12))
    nn.init.zeros_(dec[-1].weight)
    nn.init.zeros_(dec[-1].bias)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dyn.parameters()) + list(dec.parameters()), lr=3e-3)
    params = list(enc.parameters()) + list(dyn.parameters()) + list(dec.parameters())
    for _ in range(epochs):
        z = enc(st)
        zp = dyn(torch.cat([z, at], 1))
        res = dec(zp)
        dp, dth, dv, dw = res[:, 0:3], res[:, 3:6], res[:, 6:9], res[:, 9:12]
        Rn = torch.matmul(R, _exp_t(dth))
        ch = _chordal_t(Rn, R2)
        loss = (
            (p + dp - p2).norm(dim=-1) / L_REF
            + ch
            + (v + dv - v2).norm(dim=-1) / V_REF
            + (w + dw - w2).norm(dim=-1) / W_REF
        ).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    def encode(feat):
        z = np.clip((np.asarray(feat, dtype=np.float64) - mu) / sd, -30, 30).astype(np.float32)
        with torch.no_grad():
            return enc(torch.from_numpy(z)).numpy().astype(np.float64)

    def predict_res(feat, aa):
        z = np.clip((np.asarray(feat, dtype=np.float64) - mu) / sd, -30, 30).astype(np.float32)
        aa = np.clip((np.asarray(aa, dtype=np.float64) - mua) / sda, -30, 30).astype(np.float32)
        with torch.no_grad():
            zz = enc(torch.from_numpy(z))
            zp = dyn(torch.cat([zz, torch.from_numpy(aa)], 1))
            return dec(zp).numpy().astype(np.float64)

    n_param = int(sum(p.numel() for p in list(enc.parameters()) + list(dyn.parameters()) + list(dec.parameters())))
    return predict_res, encode, n_param


def _chordal_t(Ra, Rb):
    """(1-cos θ)/2 ∈ [0,1]; finite grads at 0 and π (unlike acos)."""
    import torch

    R = torch.matmul(Ra.transpose(-1, -2), Rb)
    c = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return 0.5 * (1.0 - c)


def _exp_t(w):
    import torch

    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    kx, ky, kz = k.unbind(-1)
    K = torch.stack(
        [
            torch.stack([torch.zeros_like(kx), -kz, ky], -1),
            torch.stack([kz, torch.zeros_like(kx), -kx], -1),
            torch.stack([-ky, kx, torch.zeros_like(kx)], -1),
        ],
        -2,
    )
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[0], 3, 3)
    sth = torch.sin(th)[..., None]
    cth = 1.0 - torch.cos(th)[..., None]
    return I + sth * K + cth * (K @ K)


def _step_pred(s, a, predict_res, G_in: np.ndarray | None):
    sc = dict(s)
    aa = a
    if G_in is not None:
        sc, g = canon_state(s, G_in)
        aa = transform_action(a, g)
    res = predict_res(pack_state(sc)[None], pack_action(aa)[None])[0]
    p, R, v, w = _apply_res(sc["p"][None], sc["R"][None], sc["v"][None], sc["w"][None], res[None])
    return {"p": p[0], "R": R[0], "v": v[0], "w": w[0]}


def rollout_E(trajs, acts, none, predict_res, G_in: np.ndarray | None, G_eval: np.ndarray) -> tuple[float, float]:
    es = []
    for tr, a0 in zip(trajs, acts):
        s = {k: tr[0][k].copy() for k in ("p", "R", "v", "w")}
        ds = []
        for t in range(len(tr) - 1):
            a = a0 if t == 0 else none
            s = _step_pred(s, a, predict_res, G_in)
            gt = tr[t + 1]
            ds.append(
                float(
                    d_G_np(
                        s["p"],
                        s["R"],
                        s["v"],
                        s["w"],
                        gt["p"],
                        gt["R"],
                        gt["v"],
                        gt["w"],
                        G_eval,
                    )
                )
            )
        es.append(float(np.mean(ds)))
    arr = np.asarray(es, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def tf_E(trajs, acts, none, predict_res, G_in: np.ndarray | None, G_eval: np.ndarray) -> tuple[float, float]:
    ds = []
    for tr, a0 in zip(trajs, acts):
        for t in range(len(tr) - 1):
            s = {k: tr[t][k].copy() for k in ("p", "R", "v", "w")}
            a = a0 if t == 0 else none
            sp = _step_pred(s, a, predict_res, G_in)
            gt = tr[t + 1]
            ds.append(
                float(d_G_np(sp["p"], sp["R"], sp["v"], sp["w"], gt["p"], gt["R"], gt["v"], gt["w"], G_eval))
            )
    arr = np.asarray(ds, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def persist_tf_E(trajs, G_eval: np.ndarray) -> float:
    ds = []
    for tr in trajs:
        for t in range(len(tr) - 1):
            s, gt = tr[t], tr[t + 1]
            ds.append(float(d_G_np(s["p"], s["R"], s["v"], s["w"], gt["p"], gt["R"], gt["v"], gt["w"], G_eval)))
    return float(np.mean(np.asarray(ds, dtype=np.float64)))


def _p90(curve: list[tuple[int, float]], e_ref: float) -> float:
    ok = [p for p, e in curve if np.isfinite(e) and e <= EPS_NI * e_ref]
    return float(min(ok)) if ok else float("inf")


def _pattern(*, g0: bool, g1: bool, g2: bool, g3: bool, g4: bool) -> str:
    if not g0:
        return "instrument_failure"
    if not g1:
        return "quotient_hurts_prediction"
    if not g2:
        return "discovered_ne_oracle"
    if not g3:
        return "false_compress_C4"
    if not g4:
        return "gauge_still_in_quotient"
    return "quotient_utility_supported"


def run_symx_x1(output: str | Path, config: SYMX1Config | None = None) -> dict[str, Any]:
    cfg = _lock(config or SYMX1Config())
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _refuse_r10(root)
    x0 = _require_x0(Path(cfg.x0_summary))
    cmap = _cand_map()
    bodies = make_bodies()
    header = {
        "stage": "SYM-X1",
        "schema": SCHEMA_ID,
        "prereg": PREREG_PATH,
        "host": HOST_ID,
        "no_rgb": True,
        "x0_pattern": x0["pattern"],
        "seeds": list(cfg.seeds),
        "widths": list(cfg.widths),
        "w_match": cfg.w_match,
        "config": {**asdict(cfg), "seeds": list(cfg.seeds), "widths": list(cfg.widths)},
    }
    _write_json(root / "header.json", header)
    print(f"[symx-x1] n_probe={cfg.n_probe} H={cfg.horizon} W*={cfg.w_match}", flush=True)

    per: dict[str, Any] = {}
    tr_seed, va_seed, te_seed = cfg.seeds[0], cfg.seeds[1], cfg.seeds[2]

    for cond in CONDITIONS:
        body = bodies[cond]
        Gst = _g_star(cond)
        Ghat = _g_hat(x0, cond, cmap)
        print(f"[symx-x1] {cond} |G*|={len(Gst)} |Ghat|={len(Ghat)}", flush=True)
        splits = {}
        for name, seed in (("train", tr_seed), ("val", va_seed), ("test", te_seed)):
            trajs, acts, none = _collect(body, cfg.n_probe, seed + 31 * int(cond[-1]), cfg.horizon, cfg.dt)
            splits[name] = (trajs, acts, none)
        packs = {}
        for tag, G_in in (("full", None), ("star", Gst), ("hat", Ghat)):
            packs[tag] = {sp: _transitions(*splits[sp], G_in) for sp in ("train", "val", "test")}

        e_mean = persist_tf_E(splits["test"][0], Gst)
        e_b: dict[str, dict[str, Any]] = {}

        # B0 / B1 / B2 capacity where needed
        for bid, G_in_key, G_loss, sweep in (
            ("B0", "full", np.eye(3)[None], True),
            ("B1", "star", Gst, False),
            ("B2", "hat", Ghat, True),
        ):
            widths = list(cfg.widths) if sweep else [cfg.w_match]
            curve = []
            match_pred = match_hid = match_p = None
            for W in widths:
                pred, hid, nP = _train_mlp(packs[G_in_key]["train"], hidden=W, G_loss=G_loss, epochs=cfg.epochs, seed=7 + 17 * ord(bid[-1]) + W)
                Gin = None if G_in_key == "full" else (Gst if G_in_key == "star" else Ghat)
                e, sd = tf_E(splits["test"][0], splits["test"][1], splits["test"][2], pred, Gin, Gst)
                curve.append((nP, e))
                print(f"[symx-x1] {cond} {bid} W={W} P={nP} E_tf={e:.4f}", flush=True)
                if W == cfg.w_match:
                    e_ar, _ = rollout_E(splits["test"][0], splits["test"][1], splits["test"][2], pred, Gin, Gst)
                    match_pred, match_hid, match_p, match_e, match_sd, match_ar = pred, hid, nP, e, sd, e_ar
            e_b[bid] = {
                "P": match_p,
                "E": match_e,
                "E_std": match_sd,
                "E_ar": match_ar,
                "curve": [{"P": p, "E": e} for p, e in curve],
                "pred": match_pred,
                "hid": match_hid,
                "pack_key": G_in_key,
            }

        pred3, enc3, p3 = _train_latent(packs["full"]["train"], hidden=cfg.w_match, z_dim=Z_DIM, epochs=cfg.epochs, seed=99)
        e3, sd3 = tf_E(splits["test"][0], splits["test"][1], splits["test"][2], pred3, None, Gst)
        e3_ar, _ = rollout_E(splits["test"][0], splits["test"][1], splits["test"][2], pred3, None, Gst)
        print(f"[symx-x1] {cond} B3 P={p3} E_tf={e3:.4f} E_ar={e3_ar:.4f}", flush=True)

        def probe(bid: str) -> dict[str, float]:
            if bid == "B3":
                xtr = enc3(packs["full"]["train"]["feat"])
                xte = enc3(packs["full"]["test"]["feat"])
                ytr, yte = packs["full"]["train"], packs["full"]["test"]
            else:
                key = e_b[bid]["pack_key"]
                ytr, yte = packs[key]["train"], packs[key]["test"]
                xtr = ytr["feat"]
                xte = yte["feat"]
            return {
                "yaw_R2": ridge_r2(xtr, ytr["yaw"], xte, yte["yaw"]),
                "axis_R2": axis_r2(xtr, ytr["ax"], xte, yte["ax"]),
            }

        pr = {b: probe(b) for b in ("B0", "B1", "B2", "B3")}
        # B3 encode uses full pack yaw from original R in transitions without canon — yes full pack yaw is from tr[t]['R'] original.
        # For B1/B2 packs, yaw still stored from original tr[t]['R'] before canon in _transitions — check: yaw.append(yaw_cs(tr[t]["R"])) uses ORIGINAL. Good: probe s/G feat vs true yaw.

        p90_b0 = _p90([(c["P"], c["E"]) for c in e_b["B0"]["curve"]], e_b["B0"]["E"])
        p90_b2 = _p90([(c["P"], c["E"]) for c in e_b["B2"]["curve"]], e_b["B0"]["E"])
        rp = float("nan") if not np.isfinite(p90_b0) or p90_b0 <= 0 or not np.isfinite(p90_b2) else float(1.0 - p90_b2 / p90_b0)

        few = []
        if cond == "C0" and not cfg.smoke:
            tr0, ac0, n0 = splits["train"]
            for K in FEW_K:
                sub = _transitions(tr0[:K], ac0[:K], n0, Ghat)
                pred, _, nP = _train_mlp(sub, hidden=cfg.w_match, G_loss=Ghat, epochs=cfg.epochs, seed=3 + K)
                e, _ = tf_E(splits["test"][0], splits["test"][1], splits["test"][2], pred, Ghat, Gst)
                few.append({"K": K, "P": nP, "E_B2": e})

        per[cond] = {
            "n_Gstar": int(len(Gst)),
            "n_Ghat": int(len(Ghat)),
            "E_mean": e_mean,
            "B0": {k: e_b["B0"][k] for k in ("P", "E", "E_std", "E_ar", "curve")},
            "B1": {k: e_b["B1"][k] for k in ("P", "E", "E_std", "E_ar", "curve")},
            "B2": {k: e_b["B2"][k] for k in ("P", "E", "E_std", "E_ar", "curve")},
            "B3": {"P": p3, "E": e3, "E_std": sd3, "E_ar": e3_ar, "z_dim": Z_DIM},
            "probe": pr,
            "P90_B0": p90_b0,
            "P90_B2": p90_b2,
            "R_P": rp,
            "few_shot": few,
            "ratio_B2_B0": float(e_b["B2"]["E"] / max(e_b["B0"]["E"], 1e-12)),
            "ratio_B2_B1": float(e_b["B2"]["E"] / max(e_b["B1"]["E"], 1e-12)),
            "ratio_B0_mean": float(e_b["B0"]["E"] / max(e_mean, 1e-12)),
        }
        print(
            f"[symx-x1] {cond} E0={e_b['B0']['E']:.4f} E1={e_b['B1']['E']:.4f} E2={e_b['B2']['E']:.4f} E3={e3:.4f} "
            f"mean={e_mean:.4f} R_P={rp:.3f} yawB2={pr['B2']['yaw_R2']:.3f} yawB3={pr['B3']['yaw_R2']:.3f} axB2={pr['B2']['axis_R2']:.3f}",
            flush=True,
        )

    c0, c4 = per["C0"], per["C4"]
    g0 = bool(c0["B0"]["E"] < G0_RATIO * c0["E_mean"])
    g1 = bool(c0["B2"]["E"] <= EPS_NI * c0["B0"]["E"])
    g2 = bool(c0["B2"]["E"] <= EPS_NI * c0["B1"]["E"])
    g3 = bool(c4["B2"]["E"] <= EPS_NI * c4["B0"]["E"])
    g4 = bool(c0["probe"]["B1"]["yaw_R2"] <= YAW_R2_MAX and c0["probe"]["B2"]["yaw_R2"] <= YAW_R2_MAX and c0["probe"]["B1"]["axis_R2"] >= AXIS_R2_MIN and c0["probe"]["B2"]["axis_R2"] >= AXIS_R2_MIN)
    pattern = _pattern(g0=g0, g1=g1, g2=g2, g3=g3, g4=g4)
    latent_q = bool(c0["B3"]["E"] <= EPS_NI * c0["B0"]["E"] and c0["probe"]["B3"]["yaw_R2"] <= YAW_R2_MAX)
    print(f"[symx-x1] G0={g0} G1={g1} G2={g2} G3={g3} G4={g4} pattern={pattern}", flush=True)

    summary = {
        "header": header,
        "pattern": pattern,
        "G0_fit": {"ok": g0, "E_B0": c0["B0"]["E"], "E_mean": c0["E_mean"], "ratio": c0["ratio_B0_mean"]},
        "G1_noninferior": {"ok": g1, "E_B2": c0["B2"]["E"], "E_B0": c0["B0"]["E"], "ratio": c0["ratio_B2_B0"]},
        "G2_oracle_gap": {"ok": g2, "ratio": c0["ratio_B2_B1"]},
        "G3_C4": {"ok": g3, "E_B2": c4["B2"]["E"], "E_B0": c4["B0"]["E"], "ratio": c4["ratio_B2_B0"]},
        "G4_leakage": {
            "ok": g4,
            "B1_yaw": c0["probe"]["B1"]["yaw_R2"],
            "B2_yaw": c0["probe"]["B2"]["yaw_R2"],
            "B1_axis": c0["probe"]["B1"]["axis_R2"],
            "B2_axis": c0["probe"]["B2"]["axis_R2"],
            "B3_yaw": c0["probe"]["B3"]["yaw_R2"],
            "B0_yaw": c0["probe"]["B0"]["yaw_R2"],
        },
        "latent_already_quotients": latent_q,
        "per_condition": per,
        "capacity_by_G": {c: {"n_Ghat": per[c]["n_Ghat"], "R_P": per[c]["R_P"], "P90_B0": per[c]["P90_B0"], "P90_B2": per[c]["P90_B2"]} for c in CONDITIONS},
        "unlocks_symx2_prereg": pattern == "quotient_utility_supported",
        "unlocks_symx3": False,
        "unlocks_o0g6r": False,
        "unlocks_o1": False,
    }
    _write_json(root / "summary.json", summary)
    _write_json(root / "run.json", summary)
    return summary
