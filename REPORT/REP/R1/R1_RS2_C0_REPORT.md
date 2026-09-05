# R1-RS2-C0 Report — Oracle Contact Closure and Adapter Freeze

Date: 2026-08-16  
Status: **RS2-C0 PASS; infrastructure closed; RS2 Formal implementation unlocked**  
Scientific result: **none** — this is an interface/closure stage  
Passing artifact: `runs/r1_rs2/c0_v2/summary.json`

## Verdict

\[
\boxed{
RS2\text{-C0\_PASS=true}
}
\]

Robot–Door contact generalized force is now explicit learner-visible input,
the contact force projection closes independently, normal contact no longer
appears as structural residual, and the contact excitation adapter is frozen.

This does **not** set `RS2_GO=true`. It only permits implementation of the
held-out RS2 Formal matrix.

## Frozen implementation

- backend: `aprwm_v0/r1_rs2.py`;
- CLI: `python -m aprwm_v0 r1-rs2-c0`;
- environment: `robosuite==1.5.2`, `mujoco==3.11.0`, Door + Panda;
- controller: deterministic default Panda `OSC_POSE`, 20 Hz;
- no RGB, learned policy, learned contact estimator, latch, hidden operator,
  or direct Door-hinge torque;
- simulation-rate force logging at every MuJoCo substep;
- manifest SHA256:
  `3dee8f6160e041b1a5a1499d3e011c5e62a39f378b9a21039ab6997f7e5ab89e`.

The exact Panda posture, Door pose, hinge initialization, waypoints, timing,
gripper commands, and action limits are serialized in
`runs/r1_rs2/c0_v2/script_manifest.json`.

## Contact interface

The learner receives:

\[
\tau_{\mathrm{contact},h}
=
\sum_{c\in\mathcal C_{\mathrm{robot,Door}}}
\left[(J_{p,2}-J_{p,1})^\top f_c
+(J_{r,2}-J_{r,1})^\top m_c\right]_h.
\]

`mj_contactForce` is transformed with `contact.frame.T`; the relative point
Jacobians provide the sign. A second, audit-only route projects only each
selected contact's `efc_address:efc_address+dim` rows.

Learner-visible HDF5 excludes full `qfrc_constraint`, full `qfrc_applied`, and
truth `qfrc_passive`. Declared damping and the Door-hinge
`mjCNSTR_FRICTION_DOF` row are the only nominal terms.

## C0 matrix

\[
3\text{ dev IDs}
\times3\text{ scripts}
\times3\text{ repeats}
=27\text{ cells}.
\]

The C0 geometry and Panda start posture are fixed; these are deterministic
closure replicates, not seed-level scientific samples.

| Gate | Result | Threshold |
|---|---:|---:|
| finite / complete | 27 / 27 | 27 / 27 |
| minimum Door movement | 0.11841 rad | \(\ge0.05\) |
| minimum contact-active fraction | 0.85643 | \(\ge0.10\) |
| minimum contact \(P_{95}\) | 0.39287 Nm | \(>0.02\) |
| max \(J^\top f\) vs contact-`efc` NRMSE | \(3.69\times10^{-16}\) | \(<10^{-6}\) |
| max full residual NRMSE | \(1.8879\times10^{-4}\) | \(<10^{-3}\) |
| max contact-active residual NRMSE | \(1.8877\times10^{-4}\) | \(<10^{-3}\) |
| C0 false revision | 0 / 27 | 0 |
| learner-visible leakage audit | pass | pass |

All gates passed cellwise.

## Exposure adapter freeze

Target:

\[
X_\phi(1.5A_0)=7.8029109\times10^{-4}.
\]

The preregistered grid was evaluated on development IDs only. The selected
smallest safe closest candidate is:

\[
\boxed{s^\star=2.0}
\]

with:

\[
\operatorname{median}X_\phi(u_{s^\star})
=9.2726248\times10^{-4},
\qquad
\text{relative error}=18.84\%<20\%.
\]

Frozen baseline mappings:

- `slow_pull -> 1.0A0`;
- `fast_pull -> 1.5A0`;
- `pull_release -> 1.0A0`.

No detector threshold was fitted in contact. Each baseline uses the existing
nearest frozen RS1A.5 exposure threshold.

## Development trace

The first C0 manifest used randomized reset geometry/posture and failed
contact feasibility in several dev cells. Per prereg, it was classified as an
`infrastructure/interface_block`; no formal seed was inspected. C0 v2 fixed
the scripted start geometry/posture and tuned only contact feasibility plus
the preregistered adapter. The frozen RS1C policy was not changed.

## Interpretation

The result establishes only:

\[
\boxed{
\text{normal robot contact is correctly accounted for and does not create
false structural inadequacy}
}
\]

It does not establish that RS1C remains useful under contact. That question
was reserved for the 180-episode held-out RS2 Formal run, which is now
complete: `RS2_GO=false` (`REPORT/REP/R1/R1_RS2_REPORT.md`). C0 itself is
unchanged.

## Lock state

- `RS1B_GO=false` remains unchanged;
- no RS1B.3;
- support remains monitoring-only and cannot veto;
- RS1C policy, detector, VoI, passivity, utility, and operator library remain
  frozen;
- RS2 Formal has been run once; `RS2_GO=false`;
- C0, adapter, thresholds, and policy remain frozen (no retune).
