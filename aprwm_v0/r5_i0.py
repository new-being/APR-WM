"""R5-I0: physical feasibility of local slip-margin λ. No neural probe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .mujoco_physics import import_mujoco
from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .r3_v7b3 import spearman
from .r4_c1 import FT_PROBE, PROBE_S, Y_KEYS
from .r4_i0 import (
    BLOCK_HALF_Y,
    DT,
    FINGER_HALF,
    HOLD_S,
    KD_FINGER,
    KP_FINGER,
    MU,
    PAD,
    SQUEEZE,
    TAXEL_N,
    _maps_from_contacts,
)
from .r4_i1 import HS_KEYS, _hs
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R5_I0_PREREG.md"
SOLREF_UNI = "0.022 1"
LAMBDAS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24)
LAMBDAS_V2 = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
LAMBDAS_V3 = tuple(float(x) for x in np.linspace(0.0, 0.5 * np.pi, 7))
KP_PRESTRESS = 400.0
KD_PRESTRESS = 8.0
SOLREF0 = 0.022
RHO_KT = 0.014
FT_HOLD = 0.48
D_MACRO_MAX = 0.05
RHO_MIN = 0.80
WRENCH_KEYS = ("fn", "ft", "fx", "fy", "fz", "mx", "my", "mz")
SIGNED_WRENCH_KEYS = ("fx", "fy", "fz", "mx", "my", "mz")


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mu_columns(lam: float, taxel_n: int = TAXEL_N, mu0: float = MU) -> list[float]:
    vals = []
    for ix in range(taxel_n):
        sign = 1.0 if ix < taxel_n // 2 else -1.0
        vals.append(float(mu0 + sign * lam))
    return vals


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1.0e-12))


def x_left_right(row: dict[str, Any]) -> float:
    tx = np.asarray(row["tau_x"], dtype=np.float64)
    return float(np.sum(tx[0]) - np.sum(tx[1]))


def quadrupole_modes(taxel_n: int = TAXEL_N) -> tuple[np.ndarray, np.ndarray]:
    """Mean-zero, dipole-zero pad modes (φ1=x²−z², φ2=2xz), shape (iz, ix)."""
    xs = np.linspace(-(taxel_n - 1) / 2.0, (taxel_n - 1) / 2.0, taxel_n)
    ix = xs[np.newaxis, :]
    iz = xs[:, np.newaxis]
    phi1 = ix**2 - iz**2
    phi2 = 2.0 * ix * iz
    modes = []
    for phi in (phi1, phi2):
        phi = phi - float(np.mean(phi))
        phi = phi / float(np.sqrt(np.mean(phi**2)) + 1.0e-12)
        modes.append(phi.astype(np.float64))
    return modes[0], modes[1]


def solref_timeconst_field(lam: float, taxel_n: int = TAXEL_N) -> np.ndarray:
    phi1, phi2 = quadrupole_modes(taxel_n)
    mix = np.cos(lam) * phi1 + np.sin(lam) * phi2
    peak = float(np.max(np.abs(mix))) + 1.0e-12
    field = SOLREF0 + RHO_KT * mix / peak
    return np.clip(field, 0.004, 0.040)


def mean_solref_kt(lam: float, taxel_n: int = TAXEL_N) -> float:
    return float(np.mean(solref_timeconst_field(lam, taxel_n)))


def x_stat_kt(row: dict[str, Any], taxel_n: int = TAXEL_N) -> float:
    """Projection of world-x shear onto φ2; ordered as λ: φ1 → φ2."""
    _, phi2 = quadrupole_modes(taxel_n)
    ty = np.asarray(row["tau_y"], dtype=np.float64)
    return float(np.sum(ty * phi2[np.newaxis, :, :]))


def x_stat_xhalves(row: dict[str, Any]) -> float:
    """Intra-pad x-half signed world-x shear.

    On this y-normal contact, MuJoCo's second tangent is world x, which
    R4 stores as ``tau_y``.
    """
    ty = np.asarray(row["tau_y"], dtype=np.float64)
    mid = ty.shape[2] // 2
    return float(np.sum(ty[:, :, :mid]) - np.sum(ty[:, :, mid:]))


def _wrench(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in WRENCH_KEYS], dtype=np.float64)


def _signed_wrench(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[key]) for key in SIGNED_WRENCH_KEYS], dtype=np.float64)


def _contact_row(data: Any, maps: dict[str, Any], *, q_block: int, u_finger: float, ft: float) -> dict[str, Any]:
    return {
        "t": float(data.time),
        "q": float(data.qpos[q_block]),
        "v": float(data.qvel[q_block]),
        "a": float(data.qacc[q_block]),
        "q_finger": float(data.qpos[0]),
        "v_finger": float(data.qvel[0]),
        "a_finger": float(data.qacc[0]),
        "u_finger": float(u_finger),
        "tau_motor_finger": float(data.actuator_force[0]),
        "tau_motor_block": float(data.actuator_force[q_block]),
        "ft_cmd": ft,
        "fn": maps["fn"],
        "ft": maps["ft"],
        "fx": maps["fx"],
        "fy": maps["fy"],
        "fz": maps["fz"],
        "mx": maps["mx"],
        "my": maps["my"],
        "mz": maps["mz"],
        "p": maps["p"],
        "tau_x": maps["tau_x"],
        "tau_y": maps["tau_y"],
    }


def y_future(rows: list[dict[str, Any]], start: int) -> np.ndarray:
    return np.stack(
        [[float(row[key]) for key in Y_KEYS] for row in rows[start:]],
        axis=0,
    )


def grasp_xml_prestress(*, taxel_n: int = TAXEL_N, dt: float = DT, mu0: float = MU) -> str:
    cell = PAD / taxel_n
    half = cell / 2.0 - 1.0e-4
    y_left = -(BLOCK_HALF_Y + FINGER_HALF)
    y_right = BLOCK_HALF_Y + FINGER_HALF

    def half_geoms(finger: str, y_face: float, which: str) -> str:
        ixs = range(taxel_n // 2) if which == "n" else range(taxel_n // 2, taxel_n)
        lines = []
        for ix in ixs:
            for iz in range(taxel_n):
                x = -PAD / 2.0 + (ix + 0.5) * cell
                z = -PAD / 2.0 + (iz + 0.5) * cell
                name = f"pad_{finger}_{ix}_{iz}"
                lines.append(
                    f'        <geom name="{name}" type="box" size="{half} {FINGER_HALF} {half}" '
                    f'pos="{x:.5f} {y_face:.5f} {z:.5f}" solref="{SOLREF_UNI}" '
                    f'friction="{mu0} {mu0} 0.001" condim="3" contype="1" conaffinity="2" '
                    f'rgba="0.7 0.55 0.4 1"/>'
                )
        return "\n".join(lines)

    return f"""
<mujoco model="r5_i0_prestress">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{dt}" gravity="0 0 0" integrator="Euler" cone="elliptic" impratio="1"/>
  <default>
    <joint limited="true" damping="0.08" frictionloss="0" armature="0.0002"/>
    <geom solimp="0.9 0.95 0.001" condim="3"/>
  </default>
  <worldbody>
    <body name="finger_left" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.04" diaginertia="1e-5 1e-5 1e-5"/>
      <joint name="slide_left" type="slide" axis="0 1 0" range="-0.05 0.02"/>
      <body name="left_xn">
        <joint name="left_xn" type="slide" axis="1 0 0" range="-0.004 0.004" damping="0.05"/>
{half_geoms("left", y_left, "n")}
      </body>
      <body name="left_xp">
        <joint name="left_xp" type="slide" axis="1 0 0" range="-0.004 0.004" damping="0.05"/>
{half_geoms("left", y_left, "p")}
      </body>
    </body>
    <body name="finger_right" pos="0 0 0">
      <body name="right_xn">
        <joint name="right_xn" type="slide" axis="1 0 0" range="-0.004 0.004" damping="0.05"/>
{half_geoms("right", y_right, "n")}
      </body>
      <body name="right_xp">
        <joint name="right_xp" type="slide" axis="1 0 0" range="-0.004 0.004" damping="0.05"/>
{half_geoms("right", y_right, "p")}
      </body>
    </body>
    <body name="block" pos="0 0 0">
      <joint name="slide_block" type="slide" axis="1 0 0" range="-0.08 0.08"/>
      <geom name="block" type="box" size="0.040 {BLOCK_HALF_Y} 0.040"
            mass="0.12" friction="{mu0} {mu0} 0.001" condim="3"
            contype="2" conaffinity="1" rgba="0.4 0.5 0.7 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="f_left" joint="slide_left" gear="1"/>
    <motor name="f_left_xn" joint="left_xn" gear="1"/>
    <motor name="f_left_xp" joint="left_xp" gear="1"/>
    <motor name="f_right_xn" joint="right_xn" gear="1"/>
    <motor name="f_right_xp" joint="right_xp" gear="1"/>
    <motor name="f_block" joint="slide_block" gear="1"/>
  </actuator>
  <keyframe>
    <key name="grasp" qpos="{SQUEEZE} 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def grasp_xml_margin(*, lam: float, taxel_n: int = TAXEL_N, dt: float = DT, mu0: float = MU) -> str:
    cols = mu_columns(lam, taxel_n, mu0)

    def pad_geoms(finger: str, y_face: float) -> str:
        cell = PAD / taxel_n
        half = cell / 2.0 - 1.0e-4
        lines = []
        for ix in range(taxel_n):
            mu_c = cols[ix]
            for iz in range(taxel_n):
                x = -PAD / 2.0 + (ix + 0.5) * cell
                z = -PAD / 2.0 + (iz + 0.5) * cell
                name = f"pad_{finger}_{ix}_{iz}"
                lines.append(
                    f'      <geom name="{name}" type="box" size="{half} {FINGER_HALF} {half}" '
                    f'pos="{x:.5f} {y_face:.5f} {z:.5f}" solref="{SOLREF_UNI}" '
                    f'friction="{mu_c} {mu_c} 0.001" condim="3" rgba="0.7 0.55 0.4 1"/>'
                )
        return "\n".join(lines)

    y_left = -(BLOCK_HALF_Y + FINGER_HALF)
    y_right = BLOCK_HALF_Y + FINGER_HALF
    return f"""
<mujoco model="r5_i0_margin">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{dt}" gravity="0 0 0" integrator="Euler" cone="elliptic" impratio="1"/>
  <default>
    <joint limited="true" damping="0.05" frictionloss="0" armature="0.0002"/>
    <geom solimp="0.9 0.95 0.001" condim="3"/>
  </default>
  <worldbody>
    <body name="finger_left" pos="0 0 0">
      <joint name="slide_left" type="slide" axis="0 1 0" range="-0.05 0.02"/>
{pad_geoms("left", y_left)}
    </body>
    <body name="finger_right" pos="0 0 0">
{pad_geoms("right", y_right)}
    </body>
    <body name="block" pos="0 0 0">
      <joint name="slide_block" type="slide" axis="1 0 0" range="-0.08 0.08"/>
      <geom name="block" type="box" size="0.028 {BLOCK_HALF_Y} 0.028"
            mass="0.12" friction="{mu0} {mu0} 0.001" condim="3" rgba="0.4 0.5 0.7 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="f_left" joint="slide_left" gear="1"/>
    <motor name="f_block" joint="slide_block" gear="1"/>
  </actuator>
  <keyframe>
    <key name="grasp" qpos="{SQUEEZE} 0"/>
  </keyframe>
</mujoco>
"""


def grasp_xml_kt(*, lam: float, taxel_n: int = TAXEL_N, dt: float = DT, mu0: float = MU) -> str:
    field = solref_timeconst_field(lam, taxel_n)

    def pad_geoms(finger: str, y_face: float) -> str:
        cell = PAD / taxel_n
        half = cell / 2.0 - 1.0e-4
        lines = []
        for ix in range(taxel_n):
            for iz in range(taxel_n):
                x = -PAD / 2.0 + (ix + 0.5) * cell
                z = -PAD / 2.0 + (iz + 0.5) * cell
                name = f"pad_{finger}_{ix}_{iz}"
                tau = float(field[iz, ix])
                lines.append(
                    f'      <geom name="{name}" type="box" size="{half} {FINGER_HALF} {half}" '
                    f'pos="{x:.5f} {y_face:.5f} {z:.5f}" solref="{tau:.5f} 1" '
                    f'friction="{mu0} {mu0} 0.001" condim="3" rgba="0.7 0.55 0.4 1"/>'
                )
        return "\n".join(lines)

    y_left = -(BLOCK_HALF_Y + FINGER_HALF)
    y_right = BLOCK_HALF_Y + FINGER_HALF
    return f"""
<mujoco model="r5_i0_kt">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{dt}" gravity="0 0 0" integrator="Euler" cone="elliptic" impratio="1"/>
  <default>
    <joint limited="true" damping="0.05" frictionloss="0" armature="0.0002"/>
    <geom solimp="0.9 0.95 0.001" condim="3"/>
  </default>
  <worldbody>
    <body name="finger_left" pos="0 0 0">
      <joint name="slide_left" type="slide" axis="0 1 0" range="-0.05 0.02"/>
{pad_geoms("left", y_left)}
    </body>
    <body name="finger_right" pos="0 0 0">
{pad_geoms("right", y_right)}
    </body>
    <body name="block" pos="0 0 0">
      <joint name="slide_block" type="slide" axis="1 0 0" range="-0.08 0.08"/>
      <geom name="block" type="box" size="0.028 {BLOCK_HALF_Y} 0.028"
            mass="0.12" friction="{mu0} {mu0} 0.001" condim="3" rgba="0.4 0.5 0.7 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="f_left" joint="slide_left" gear="1"/>
    <motor name="f_block" joint="slide_block" gear="1"/>
  </actuator>
  <keyframe>
    <key name="grasp" qpos="{SQUEEZE} 0"/>
  </keyframe>
</mujoco>
"""


def simulate_margin(lam: float) -> dict[str, Any]:
    mujoco = import_mujoco()
    xml = grasp_xml_margin(lam=lam)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    q_left0 = float(data.qpos[0])
    rows = []
    for step in range(n_hold + n_probe):
        ft = 0.0 if step < n_hold else FT_PROBE
        u_finger = KP_FINGER * (q_left0 - float(data.qpos[0])) - KD_FINGER * float(data.qvel[0])
        data.ctrl[0] = u_finger
        data.ctrl[1] = ft
        mujoco.mj_step(model, data)
        maps = _maps_from_contacts(mujoco, model, data, TAXEL_N)
        rows.append(_contact_row(data, maps, q_block=1, u_finger=u_finger, ft=ft))
    return _trace_pack(lam, xml, rows, n_hold, x_fn=x_left_right)


def simulate_prestress(lam: float) -> dict[str, Any]:
    mujoco = import_mujoco()
    xml = grasp_xml_prestress()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    q_left0 = float(data.qpos[0])
    rows = []
    for step in range(n_hold + n_probe):
        ft = 0.0 if step < n_hold else FT_PROBE
        u_finger = KP_FINGER * (q_left0 - float(data.qpos[0])) - KD_FINGER * float(data.qvel[0])
        data.ctrl[0] = u_finger
        signs = (1.0, -1.0, 1.0, -1.0)
        for i, sign in enumerate(signs):
            qh = float(data.qpos[1 + i])
            vh = float(data.qvel[1 + i])
            data.ctrl[1 + i] = sign * lam - KP_PRESTRESS * qh - KD_PRESTRESS * vh
        data.ctrl[5] = ft
        mujoco.mj_step(model, data)
        maps = _maps_from_contacts(mujoco, model, data, TAXEL_N)
        row = _contact_row(data, maps, q_block=5, u_finger=u_finger, ft=ft)
        row["q_xn"] = float(data.qpos[1])
        row["q_xp"] = float(data.qpos[2])
        rows.append(row)
    return _trace_pack(lam, xml, rows, n_hold, x_fn=x_stat_xhalves)


def simulate_kt(lam: float) -> dict[str, Any]:
    mujoco = import_mujoco()
    xml = grasp_xml_kt(lam=lam)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    n_hold = int(round(HOLD_S / DT))
    n_probe = int(round(PROBE_S / DT))
    q_left0 = float(data.qpos[0])
    rows = []
    for step in range(n_hold + n_probe):
        ft = FT_HOLD if step < n_hold else FT_PROBE
        u_finger = KP_FINGER * (q_left0 - float(data.qpos[0])) - KD_FINGER * float(data.qvel[0])
        data.ctrl[0] = u_finger
        data.ctrl[1] = ft
        mujoco.mj_step(model, data)
        maps = _maps_from_contacts(mujoco, model, data, TAXEL_N)
        rows.append(_contact_row(data, maps, q_block=1, u_finger=u_finger, ft=ft))
    pack = _trace_pack(lam, xml, rows, n_hold, x_fn=x_stat_kt)
    ty = np.asarray(rows[n_hold - 1]["tau_y"], dtype=np.float64)
    pack["tau_sign_frac"] = float(np.mean(ty > 0.0)) if float(np.sum(np.abs(ty))) > 0 else 0.0
    pack["mean_solref"] = mean_solref_kt(lam)
    return pack


def _trace_pack(
    lam: float,
    xml: str,
    rows: list[dict[str, Any]],
    n_hold: int,
    *,
    x_fn,
) -> dict[str, Any]:
    t_star = n_hold - 1
    star = rows[t_star]
    return {
        "lam": float(lam),
        "t_star": t_star,
        "hs": _hs(star),
        "wrench": _wrench(star),
        "signed_wrench": _signed_wrench(star),
        "x_stat": x_fn(star),
        "x_lr": x_left_right(star),
        "x_halves": x_stat_xhalves(star),
        "y_future": y_future(rows, t_star),
        "q_end": float(rows[-1]["q"]),
        "p_sum": float(np.sum(star["p"])),
        "tau_x_abs": float(np.sum(np.abs(star["tau_x"]))),
        "tau_y_abs": float(np.sum(np.abs(star["tau_y"]))),
        "fn_star": float(star["fn"]),
        "ft_star": float(star["ft"]),
        "fx_star": float(star["fx"]),
        "fy_star": float(star["fy"]),
        "mz_star": float(star["mz"]),
        "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
    }


def _scan(
    traces: list[dict[str, Any]],
    lambdas: tuple[float, ...],
    *,
    mechanism: str,
    pass_key: str,
) -> dict[str, Any]:
    hs0 = traces[0]["hs"]
    w0 = traces[0]["wrench"]
    y0 = traces[0]["y_future"]
    lams = np.asarray(lambdas, dtype=np.float64)
    d_hs = np.asarray([relative_l2(item["hs"], hs0) for item in traces])
    d_w = np.asarray([relative_l2(item["wrench"], w0) for item in traces])
    sw0 = traces[0]["signed_wrench"]
    d_sw = np.asarray([relative_l2(item["signed_wrench"], sw0) for item in traces])
    x_stat = np.asarray([item["x_stat"] for item in traces])
    c_y = np.asarray([relative_l2(item["y_future"], y0) for item in traces])
    rho_x = spearman(lams, x_stat)
    rho_c = spearman(lams, c_y)
    g_macro = bool(float(np.max(d_hs)) < D_MACRO_MAX)
    g_local = bool(rho_x > RHO_MIN)
    g_cons = bool(rho_c > RHO_MIN)
    passed = bool(g_macro and g_local and g_cons)
    status = stage_status()
    status["R4"] = {"family_stop": True}
    status["R5-I0"] = {
        "mechanism": mechanism,
        pass_key: passed,
        "g_macro": g_macro,
        "g_local": g_local,
        "g_cons": g_cons,
    }
    return {
        "stage": "R5-I0",
        "scientific_result": True,
        "neural_probe": False,
        "mechanism": mechanism,
        "prereg_sha256": _prereg_sha256(),
        "lambdas": list(lambdas),
        "d_hs": d_hs.tolist(),
        "d_wrench": d_w.tolist(),
        "d_signed_wrench": d_sw.tolist(),
        "x_stat": x_stat.tolist(),
        "x_lr": [item["x_lr"] for item in traces],
        "x_halves": [item["x_halves"] for item in traces],
        "c_future": c_y.tolist(),
        "p_sum": [item["p_sum"] for item in traces],
        "tau_x_abs": [item["tau_x_abs"] for item in traces],
        "tau_y_abs": [item["tau_y_abs"] for item in traces],
        "fn_star": [item["fn_star"] for item in traces],
        "ft_star": [item["ft_star"] for item in traces],
        "fx_star": [item["fx_star"] for item in traces],
        "fy_star": [item["fy_star"] for item in traces],
        "mz_star": [item["mz_star"] for item in traces],
        "max_d_hs": float(np.max(d_hs)),
        "max_d_wrench": float(np.max(d_w)),
        "max_d_signed_wrench": float(np.max(d_sw)),
        "spearman_lam_x": rho_x,
        "spearman_lam_c": rho_c,
        "d_macro_max": D_MACRO_MAX,
        "rho_min": RHO_MIN,
        "g_macro": g_macro,
        "g_local": g_local,
        "g_cons": g_cons,
        pass_key: passed,
        "r5_i0_pass": passed,
        "note": "no learner; three physical orderings only; t_star=hold end",
        "stage_status": status,
    }


def run_r5_i0_v1(output: str | Path) -> dict[str, Any]:
    traces = [simulate_margin(lam) for lam in LAMBDAS]
    payload = _scan(traces, LAMBDAS, mechanism="local_slip_margin", pass_key="r5_i0_pass")
    payload["x_lr"] = [item["x_lr"] for item in traces]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload


def run_r5_i0(output: str | Path, *, mechanism: str = "kt") -> dict[str, Any]:
    if mechanism == "margin":
        return run_r5_i0_v1(output)
    if mechanism == "prestress":
        traces = [simulate_prestress(lam) for lam in LAMBDAS_V2]
        payload = _scan(
            traces,
            LAMBDAS_V2,
            mechanism="self_equilibrated_prestress",
            pass_key="r5_i0_v2_pass",
        )
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "summary.json", _jsonable(payload))
        return payload
    traces = [simulate_kt(lam) for lam in LAMBDAS_V3]
    payload = _scan(
        traces,
        LAMBDAS_V3,
        mechanism="integral_constrained_kt",
        pass_key="r5_i0_v3_pass",
    )
    payload["ft_hold"] = FT_HOLD
    payload["ft_probe"] = FT_PROBE
    payload["rho_kt"] = RHO_KT
    payload["solref0"] = SOLREF0
    payload["mean_solref"] = [item["mean_solref"] for item in traces]
    payload["tau_sign_frac"] = [item["tau_sign_frac"] for item in traces]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    return payload
