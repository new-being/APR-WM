# R10-C0 residual.h5 schema

Date: 2026-08-17  
Status: **FROZEN contract**; does not unlock C0  
Path: `runs/r10_c0/real/residual.h5`  
Not valid: R1-MJ0 / RS0 / MuJoCo truth dumps copied into this path

\[
\boxed{
\text{real hardware residual.h5}
\to
\text{C0 gate}
}
\]

Not: simulator residual \(\to\) C0.

## Root attributes

| attr | type | meaning |
|---|---|---|
| `schema` | string | must be `aprwm.r10_c0.residual.v1` |
| `source` | string | `hardware` or `device_log` only |
| `platform` | string | chosen **before** collection |
| `shadow_only` | bool | must be true |
| `autonomous_revision` | bool | must be false |
| `dt` | float | physics / joint-torque sample period (s) |
| `n_dof` | int | generalized-force dimension |
| `qacc_estimator` | string | how \(\ddot q\) is obtained (filter name + params) |
| `qvel_estimator` | string | how \(\dot q\) is obtained (not a hidden low-rate channel) |
| `tau_source` | string | H1: `inline_rotary_transducer` |
| `timebase` | string | `aligned` if \(t_q=t_\tau=t_{\mathrm{bus}}\), else `offset` |
| `dt_offset_q_tau` | float | DC target \(\approx 0\) |
| `dt_offset_u_tau` | float | actuator delay; **not** assumed 0 because of CST |
| `dt_offset_q_u` | float | optional; prefer \(u\) on the same bus cycle |

`source=simulator` is invalid. Copying an MJ0/RS0 cell into this path
is invalid.

## Groups / datasets

All series `(T, n_dof)` float64 unless noted. Rate is **joint-torque /
physics**, not 20 Hz policy.

| path | required | notes |
|---|---|---|
| `t` | yes | `(T,)` seconds, common timebase |
| `learner_visible/qpos` | yes | **hinge-side** \(q_{\mathrm{plant}}\) |
| `diagnostics/motor_qpos` | yes on H1 | rotor encoder; not the C0 \(q\) |
| `learner_visible/qvel` | yes | measured or filtered \(\dot q\) |
| `learner_visible/qacc` | yes | estimated \(\ddot q\); method in `qacc_estimator` |
| `learner_visible/tau_meas` | yes | measured torque |
| `learner_visible/tau_nominal` | yes | frozen nominal; H1 uses sensing-plane \(I_{\mathrm{CAD}}\) (`R10_C0_H1_CAD0.md`) |
| `learner_visible/residual` | yes | \(r=\tau_{\mathrm{meas}}-\tau_{\mathrm{nominal}}\) |
| `learner_visible/u_cmd` | yes | `(T, n_u)` executed command; for later \(I(\mathcal V;Y\mid a)\) |
| `metadata/json` | yes | sensors, cal date, filters, safety limits |
| `evaluation/nrmse` | no | written by `r10-c0` after a preregistered floor exists |

## Residual definition

\[
r_{\mathrm{real}}=\tau_{\mathrm{measured}}-\tau_{\mathrm{nominal}}.
\]

Attributable physics mismatch on the real chain, not a scalar error
without \(\tau_{\mathrm{nominal}}\).

Must **not** mix in `qfrc_passive_truth` or MuJoCo `qfrc_bias` as
ground truth. Constraint / contact estimates, if logged, live only
under `diagnostics/` and are **excluded** from C0 residual by default.

## C0 gate (after a valid file exists)

No MJ0 threshold \(10^{-4}\).

1. First hardware **noise characterization** (nominal, this schema).
2. Preregister \(\mathrm{NRMSE}_{\mathrm{floor}}\) from that campaign.
3. Then run the C0 gate against that floor.

Until `residual.h5` exists, `r10-c0` rejects. First logged trajectory
answers only: **what is the real residual floor?** It does not prove
APR-WM.
