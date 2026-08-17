"""R4-I0 v2: same μ / preload; left–right compliance swap.

Minimal success: matched global mechanics and different local traction.
Does not evaluate R4-C0. Does not train a probe.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .mujoco_physics import import_mujoco
from .r1_mj import stage_status
from .r1_rs3a import _jsonable
from .v06 import _write_json


PREREG_PATH = "REPORT/REG/R4_I0_PREREG.md"
TAXEL_N = 4
PAD = 0.036
BLOCK_HALF_Y = 0.012
FINGER_HALF = 0.006
DT = 0.002
MU = 0.80
FT_PEAK = 2.4
HOLD_S = 0.20
RAMP_S = 0.60
POST_S = 0.15
SQUEEZE = 0.0025
KP_FINGER = 300.0
KD_FINGER = 10.0
SOLREF_STIFF = "0.004 1"
SOLREF_SOFT = "0.040 1"
EPS_Q = 1.0e-3
EPS_V = 5.0e-3
EPS_N = 0.20
EPS_T = 0.20
EPS_U = 0.25
D_TAXEL_MIN = 0.08
N_MATCH_MIN = 8
Pattern = Literal["stiff_left", "stiff_right"]


def _prereg_sha256() -> str:
    path = Path(PREREG_PATH)
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class R4I0Config:
    taxel_n: int = TAXEL_N
    dt: float = DT
    mu: float = MU
    ft_peak: float = FT_PEAK
    hold_s: float = HOLD_S
    ramp_s: float = RAMP_S
    post_s: float = POST_S
    protocol: str = "stiffness_swap_v2"


def _solref_for_column(ix: int, taxel_n: int, pattern: Pattern) -> str:
    left = ix < taxel_n // 2
    stiff_on_left = pattern == "stiff_left"
    if left == stiff_on_left:
        return SOLREF_STIFF
    return SOLREF_SOFT


def grasp_xml(*, mu: float, pattern: Pattern, taxel_n: int = TAXEL_N, dt: float = DT) -> str:
    def pad_geoms(finger: str, y_face: float) -> str:
        cell = PAD / taxel_n
        half = cell / 2.0 - 1.0e-4
        lines = []
        for ix in range(taxel_n):
            solref = _solref_for_column(ix, taxel_n, pattern)
            for iz in range(taxel_n):
                x = -PAD / 2.0 + (ix + 0.5) * cell
                z = -PAD / 2.0 + (iz + 0.5) * cell
                name = f"pad_{finger}_{ix}_{iz}"
                lines.append(
                    f'      <geom name="{name}" type="box" size="{half} {FINGER_HALF} {half}" '
                    f'pos="{x:.5f} {y_face:.5f} {z:.5f}" solref="{solref}" '
                    f'friction="{mu} {mu} 0.001" condim="3" rgba="0.7 0.55 0.4 1"/>'
                )
        return "\n".join(lines)

    y_left = -(BLOCK_HALF_Y + FINGER_HALF)
    y_right = BLOCK_HALF_Y + FINGER_HALF
    return f"""
<mujoco model="r4_i0_swap">
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
            mass="0.12" friction="{mu} {mu} 0.001" condim="3" rgba="0.4 0.5 0.7 1"/>
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


def mean_solref_timeconst(pattern: Pattern, taxel_n: int = TAXEL_N) -> float:
    vals = []
    for ix in range(taxel_n):
        token = _solref_for_column(ix, taxel_n, pattern).split()[0]
        vals.extend([float(token)] * taxel_n)
    return float(np.mean(vals))


def _taxel_index(name: str, taxel_n: int) -> tuple[int, int, int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "pad":
        return None
    finger = 0 if parts[1] == "left" else 1
    return finger, int(parts[2]), int(parts[3])


def _maps_from_contacts(mujoco: Any, model: Any, data: Any, taxel_n: int) -> dict[str, Any]:
    pressure = np.zeros((2, taxel_n, taxel_n), dtype=np.float64)
    tau_x = np.zeros_like(pressure)
    tau_y = np.zeros_like(pressure)
    fn = 0.0
    ft = 0.0
    fx = fy = fz = 0.0
    mx = my = mz = 0.0
    block_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block"))
    block_pos = np.asarray(data.xpos[block_id], dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""
            for gid in (int(contact.geom1), int(contact.geom2))
        ]
        pad = next((n for n in names if n.startswith("pad_")), None)
        if pad is None:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, wrench)
        p = abs(float(wrench[0]))
        tx, ty = float(wrench[1]), float(wrench[2])
        fn += p
        ft += math.hypot(tx, ty)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame.T @ wrench[:3]
        fx += float(force_world[0])
        fy += float(force_world[1])
        fz += float(force_world[2])
        r = np.asarray(contact.pos, dtype=np.float64) - block_pos
        moment = np.cross(r, force_world)
        mx += float(moment[0])
        my += float(moment[1])
        mz += float(moment[2])
        loc = _taxel_index(pad, taxel_n)
        if loc is None:
            continue
        finger, ix, iz = loc
        pressure[finger, iz, ix] += p
        tau_x[finger, iz, ix] += tx
        tau_y[finger, iz, ix] += ty
    return {
        "p": pressure,
        "tau_x": tau_x,
        "tau_y": tau_y,
        "fn": fn,
        "ft": ft,
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "mx": mx,
        "my": my,
        "mz": mz,
        "centroid_tau": _centroid(np.abs(tau_x)),
    }


def _centroid(mass_map: np.ndarray) -> float:
    mass = float(np.sum(mass_map))
    if mass < 1.0e-9:
        return 0.0
    nx = mass_map.shape[2]
    xs = np.linspace(-0.5, 0.5, nx)
    weights = mass_map.sum(axis=(0, 1))
    return float(np.dot(weights, xs) / (float(np.sum(weights)) + 1.0e-12))


def d_taxel(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(
        np.linalg.norm(np.asarray(a["tau_x"]) - np.asarray(b["tau_x"]))
        + np.linalg.norm(np.asarray(a["tau_y"]) - np.asarray(b["tau_y"]))
        + 0.25 * np.linalg.norm(np.asarray(a["p"]) - np.asarray(b["p"]))
    )


def mechanics_matched(a: dict[str, Any], b: dict[str, Any]) -> bool:
    fn_bar = 0.5 * (abs(a["fn"]) + abs(b["fn"])) + 1.0e-6
    ft_bar = 0.5 * (abs(a["ft"]) + abs(b["ft"])) + 1.0e-6
    u_bar = 0.5 * (abs(a["u_finger"]) + abs(b["u_finger"])) + 1.0e-6
    return bool(
        abs(a["q"] - b["q"]) < EPS_Q
        and abs(a["v"] - b["v"]) < EPS_V
        and abs(a["fn"] - b["fn"]) / fn_bar < EPS_N
        and abs(a["ft"] - b["ft"]) / ft_bar < EPS_T
        and abs(a["u_finger"] - b["u_finger"]) / u_bar < EPS_U
        and abs(a["ft_cmd"] - b["ft_cmd"]) < 1.0e-9
        and abs(a["q_finger"] - b["q_finger"]) < EPS_Q
    )


def simulate_pattern(
    pattern: Pattern,
    config: R4I0Config,
    *,
    q0_block: float = 0.0,
    probe_ft: float | None = None,
    probe_s: float = 0.40,
) -> dict[str, Any]:
    mujoco = import_mujoco()
    xml = grasp_xml(mu=config.mu, pattern=pattern, taxel_n=config.taxel_n, dt=config.dt)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[1] = float(q0_block)
    mujoco.mj_forward(model, data)
    n_hold = int(round(config.hold_s / config.dt))
    n_ramp = int(round(config.ramp_s / config.dt))
    n_post = int(round(config.post_s / config.dt))
    n_probe = int(round(probe_s / config.dt)) if probe_ft is not None else 0
    n_total = n_hold + n_probe if probe_ft is not None else n_hold + n_ramp + n_post
    q_left0 = float(data.qpos[0])
    rows = []
    for step in range(n_total):
        if probe_ft is not None:
            ft = 0.0 if step < n_hold else float(probe_ft)
        elif step < n_hold:
            ft = 0.0
        elif step < n_hold + n_ramp:
            frac = (step - n_hold) / max(n_ramp - 1, 1)
            ft = config.ft_peak * frac
        else:
            ft = config.ft_peak
        u_finger = KP_FINGER * (q_left0 - float(data.qpos[0])) - KD_FINGER * float(data.qvel[0])
        data.ctrl[0] = u_finger
        data.ctrl[1] = ft
        mujoco.mj_step(model, data)
        maps = _maps_from_contacts(mujoco, model, data, config.taxel_n)
        rows.append(
            {
                "t": float(data.time),
                "q": float(data.qpos[1]),
                "v": float(data.qvel[1]),
                "a": float(data.qacc[1]),
                "q_finger": float(data.qpos[0]),
                "v_finger": float(data.qvel[0]),
                "a_finger": float(data.qacc[0]),
                "u_finger": float(u_finger),
                "tau_motor_finger": float(data.actuator_force[0]),
                "tau_motor_block": float(data.actuator_force[1]),
                "ft_cmd": ft,
                "fn": maps["fn"],
                "ft": maps["ft"],
                "fx": maps["fx"],
                "fy": maps["fy"],
                "fz": maps["fz"],
                "mx": maps["mx"],
                "my": maps["my"],
                "mz": maps["mz"],
                "centroid_tau": maps["centroid_tau"],
                "p": maps["p"],
                "tau_x": maps["tau_x"],
                "tau_y": maps["tau_y"],
            }
        )
    return {
        "pattern": pattern,
        "q0_block": float(q0_block),
        "mu": config.mu,
        "xml_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "n_steps": len(rows),
        "n_hold": n_hold,
        "fn_mean": float(np.mean([row["fn"] for row in rows])),
        "ft_max": float(max(row["ft"] for row in rows)),
        "q_abs_max": float(max(abs(row["q"]) for row in rows)),
        "map_l1": float(sum(np.sum(np.abs(row["p"])) + np.sum(np.abs(row["tau_x"])) for row in rows)),
        "mean_solref": mean_solref_timeconst(pattern, config.taxel_n),
        "rows": rows,
    }


def _i0_gates(pat_a: dict[str, Any], pat_b: dict[str, Any]) -> dict[str, Any]:
    maps_ok = pat_a["map_l1"] > 1.0 and pat_b["map_l1"] > 1.0
    solref_matched = abs(pat_a["mean_solref"] - pat_b["mean_solref"]) < 1.0e-12
    mu_matched = abs(pat_a["mu"] - pat_b["mu"]) < 1.0e-12
    n_macro = 0
    n_counter = 0
    d_on_macro: list[float] = []
    d_centroid: list[float] = []
    for a, b in zip(pat_a["rows"], pat_b["rows"], strict=True):
        if not mechanics_matched(a, b):
            continue
        n_macro += 1
        dist = d_taxel(a, b)
        d_on_macro.append(dist)
        d_centroid.append(abs(a["centroid_tau"] - b["centroid_tau"]))
        if dist > D_TAXEL_MIN:
            n_counter += 1
    gate0 = n_macro >= N_MATCH_MIN
    mean_d = float(np.mean(d_on_macro)) if d_on_macro else 0.0
    gate1 = n_counter >= N_MATCH_MIN and mean_d > D_TAXEL_MIN
    return {
        "maps_nonzero": maps_ok,
        "mu_matched": mu_matched,
        "mean_solref_matched": solref_matched,
        "n_macro_matched": n_macro,
        "n_counterexample": n_counter,
        "mean_d_taxel_on_macro": mean_d,
        "mean_centroid_gap_on_macro": float(np.mean(d_centroid)) if d_centroid else 0.0,
        "d_taxel_min": D_TAXEL_MIN,
        "gate0_macro_matched": gate0,
        "gate1_local_split": gate1,
        "r4_i0_pass": bool(maps_ok and mu_matched and solref_matched and gate0 and gate1),
    }


def _slim(trace: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if key != "rows"}


def run_r4_i0(output: str | Path, *, config: R4I0Config | None = None) -> dict[str, Any]:
    cfg = config or R4I0Config()
    pat_a = simulate_pattern("stiff_left", cfg)
    pat_b = simulate_pattern("stiff_right", cfg)
    gates = _i0_gates(pat_a, pat_b)
    status = stage_status()
    status["R3-V7"] = {"frozen": True, "closed": True, "no_v7e": True}
    status["R4-I0"] = {
        "protocol": "stiffness_swap_v2",
        "r4_i0_pass": gates["r4_i0_pass"],
        "v1_mu_mismatch": "FAIL retained",
    }
    status["R4-I1"] = {"locked": not gates["r4_i0_pass"]}
    status["R4-C0"] = {"locked": True, "reason": "needs I0 and I1"}
    payload = {
        "stage": "R4-I0",
        "protocol": "stiffness_swap_v2",
        "scientific_result": False,
        "prereg_sha256": _prereg_sha256(),
        "config": asdict(cfg),
        "pattern_a": _slim(pat_a),
        "pattern_b": _slim(pat_b),
        "gates": gates,
        "r4_i0_pass": gates["r4_i0_pass"],
        "stage_status": status,
        "note": "v2: no classifier; Gate0 macro match then Gate1 local split",
    }
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary.json", _jsonable(payload))
    windows = []
    for a, b in zip(pat_a["rows"], pat_b["rows"], strict=True):
        if not mechanics_matched(a, b):
            continue
        dist = d_taxel(a, b)
        if dist <= D_TAXEL_MIN:
            continue
        keys = ("q", "v", "fn", "ft", "q_finger", "u_finger", "ft_cmd", "p", "tau_x", "tau_y", "centroid_tau")
        windows.append({"t": a["t"], "d_taxel": dist, "a": {k: a[k] for k in keys}, "b": {k: b[k] for k in keys}})
    _write_json(root / "matched_windows.json", _jsonable({"n": len(windows), "windows": windows}))
    return payload
