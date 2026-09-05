"""R5-I0-SELFSTRESS: hidden preload → geometric stiffness. No neural probe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .mujoco_physics import import_mujoco
from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r3_v7b3 import spearman
from .r5_i0 import relative_l2
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R5/R5_I0_SELFSTRESS_PREREG.md"
DT = 0.002
HOLD_S = 0.30
PROBE_S = 0.40
HALF_SPAN = 0.08
K_TENDON = 200.0
K0_LATERAL = 20.0
MASS = 0.20
F_DIAG = 1.0
LAMBDAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
D_MACRO_MAX = 0.05
RHO_MIN = 0.80
C_RANGE_MIN = 0.10
HS_KEYS = ("q", "v", "a", "u", "tau_motor", "u_cmd")
Y_KEYS = ("q", "v", "a", "u", "tau_motor")


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rest_length(lam: float) -> float:
    l0 = HALF_SPAN - float(lam) / K_TENDON
    if l0 <= 1.0e-4:
        raise ValueError(f"rest length non-positive for lambda={lam}")
    return l0


def _hs(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in HS_KEYS], dtype=np.float64)


def grasp_xml_selfstress(*, lam: float, dt: float = DT) -> str:
    l0 = rest_length(lam)
    return f"""
<mujoco model="r5_i0_selfstress">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{dt}" gravity="0 0 0" integrator="Euler"/>
  <worldbody>
    <site name="anchor_L" pos="-{HALF_SPAN} 0 0" size="0.003"/>
    <site name="anchor_R" pos="{HALF_SPAN} 0 0" size="0.003"/>
    <body name="slider" pos="0 0 0">
      <joint name="slide_y" type="slide" axis="0 1 0" range="-0.05 0.05"
             stiffness="{K0_LATERAL}" damping="1.5" armature="0.0002"/>
      <geom name="slider" type="sphere" size="0.012" mass="{MASS}"
            contype="0" conaffinity="0" rgba="0.3 0.5 0.8 1"/>
      <site name="mass" pos="0 0 0" size="0.004"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="tendon_L" stiffness="{K_TENDON}" damping="0.2"
             springlength="{l0:.8f}" width="0.002" rgba="0.8 0.3 0.2 1">
      <site site="anchor_L"/>
      <site site="mass"/>
    </spatial>
    <spatial name="tendon_R" stiffness="{K_TENDON}" damping="0.2"
             springlength="{l0:.8f}" width="0.002" rgba="0.2 0.6 0.3 1">
      <site site="anchor_R"/>
      <site site="mass"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="f_y" joint="slide_y" gear="1"/>
  </actuator>
  <keyframe>
    <key name="center" qpos="0"/>
  </keyframe>
</mujoco>
"""


def _tendon_tension(model: Any, data: Any, index: int) -> float:
    length = float(data.ten_length[index])
    l0 = float(model.tendon_lengthspring[index, 0])
    stiff = float(model.tendon_stiffness[index])
    return stiff * (length - l0)


def simulate_selfstress(lam: float, *, f_diag: float = F_DIAG) -> dict[str, Any]:
    mujoco = import_mujoco()
    xml = grasp_xml_selfstress(lam=lam)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    rows = []
    for step in range(n_hold + n_probe):
        u_cmd = 0.0 if step < n_hold else float(f_diag)
        data.ctrl[0] = u_cmd
        mujoco.mj_step(model, data)
        t_l = _tendon_tension(model, data, 0)
        t_r = _tendon_tension(model, data, 1)
        rows.append(
            {
                "t": float(data.time),
                "q": float(data.qpos[0]),
                "v": float(data.qvel[0]),
                "a": float(data.qacc[0]),
                "u": float(data.ctrl[0]),
                "u_cmd": u_cmd,
                "tau_motor": float(data.actuator_force[0]),
                "t_left": t_l,
                "t_right": t_r,
                "x_stat": 0.5 * (t_l + t_r),
            }
        )
    t_star = n_hold - 1
    star = rows[t_star]
    y_future = np.stack(
        [[float(row[key]) for key in Y_KEYS] for row in rows[t_star:]],
        axis=0,
    )
    return {
        "lam": float(lam),
        "t_star": t_star,
        "hs": _hs(star),
        "x_stat": float(star["x_stat"]),
        "t_left": float(star["t_left"]),
        "t_right": float(star["t_right"]),
        "q_star": float(star["q"]),
        "u_star": float(star["u"]),
        "tau_star": float(star["tau_motor"]),
        "q_end": float(rows[-1]["q"]),
        "y_future": y_future,
        "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
    }


def run_r5_i0_selfstress(output: str | Path) -> dict[str, Any]:
    traces = [simulate_selfstress(lam) for lam in LAMBDAS]
    hs0 = traces[0]["hs"]
    y0 = traces[0]["y_future"]
    lams = np.asarray(LAMBDAS, dtype=np.float64)
    d_hs = np.asarray([relative_l2(item["hs"], hs0) for item in traces])
    x_stat = np.asarray([item["x_stat"] for item in traces])
    c_y = np.asarray([relative_l2(item["y_future"], y0) for item in traces])
    rho_x = spearman(lams, x_stat)
    rho_c = spearman(lams, c_y)
    rho_xc = spearman(x_stat, c_y)
    c_range = float(np.max(c_y) - np.min(c_y))
    g_macro = bool(float(np.max(d_hs)) < D_MACRO_MAX)
    g_local = bool(rho_x > RHO_MIN)
    g_cons = bool(rho_c > RHO_MIN)
    g_range = bool(c_range > C_RANGE_MIN)
    g_xc = bool(rho_xc > RHO_MIN)
    passed = bool(g_macro and g_local and g_cons and g_range and g_xc)
    status = stage_status()
    status["R4"] = {"family_stop": True}
    status["R5-contact"] = {"paused": True}
    status["R5-I0-SELFSTRESS"] = {
        "r5_i0_selfstress_pass": passed,
        "g_macro": g_macro,
        "g_local": g_local,
        "g_cons": g_cons,
        "g_range": g_range,
        "g_xc": g_xc,
    }
    payload = {
        "stage": "R5-I0-SELFSTRESS",
        "scientific_result": True,
        "neural_probe": False,
        "family": "selfstress_geometric_stiffness",
        "prereg_sha256": _prereg_sha256(),
        "lambdas": list(LAMBDAS),
        "d_hs": d_hs.tolist(),
        "x_stat": x_stat.tolist(),
        "c_future": c_y.tolist(),
        "t_left": [item["t_left"] for item in traces],
        "t_right": [item["t_right"] for item in traces],
        "q_star": [item["q_star"] for item in traces],
        "u_star": [item["u_star"] for item in traces],
        "tau_star": [item["tau_star"] for item in traces],
        "q_end": [item["q_end"] for item in traces],
        "max_d_hs": float(np.max(d_hs)),
        "c_range": c_range,
        "c_var": float(np.var(c_y)),
        "spearman_lam_x": rho_x,
        "spearman_lam_c": rho_c,
        "spearman_x_c": rho_xc,
        "d_macro_max": D_MACRO_MAX,
        "rho_min": RHO_MIN,
        "c_range_min": C_RANGE_MIN,
        "f_diag": F_DIAG,
        "k0_lateral": K0_LATERAL,
        "k_tendon": K_TENDON,
        "g_macro": g_macro,
        "g_local": g_local,
        "g_cons": g_cons,
        "g_range": g_range,
        "g_xc": g_xc,
        "r5_i0_selfstress_pass": passed,
        "note": "no learner; hidden rest-length preload; identical external ctrl",
        "stage_status": status,
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
