# REAL-LOG-A0 Report — Public Real-Robot Log Feasibility Audit

Date: 2026-08-17  
Status: **`A0-LIMITED` (aggregate)**; **not force-auditable**; **not R10**  
Method: documentation / published schema audit only — **no full download,
no training**  
Does not: unlock R10-C0; claim real-physics validation; invent \(\tau\)

## Verdict

\[
\boxed{\textbf{A0-LIMITED: state-action only}}
\]

Neither **official DROID** nor **AgiBot World (Alpha/Beta docs)** clears
**A0-GO: force-auditable** for

\[
r=\tau_{\mathrm{meas}}-\tau_{\mathrm{nominal}}.
\]

Therefore **REAL-LOG-A1 (force residual)** is **not** unlocked.
Representation / action-distribution audits remain optional and separate.

\[
\boxed{
\textbf{without controllable hardware, the bottleneck is the data source,
not another algorithm cell}
}
\]

## Audit method

Sources (primary):

| dataset | primary schema source |
|---|---|
| DROID | Official docs RLDS schema + paper Appendix B |
| AgiBot World | HuggingFace `AgiBotWorld-Beta` / Alpha README proprio tables |

No full RLDS/HDF5 download. No model training.

## DROID

### Present (documented)

| field | status |
|---|---|
| \(q\) | yes — `observation.joint_position` (7D) in official RLDS |
| EE / gripper | yes — Cartesian + gripper in RLDS |
| \(u\) | yes — policy/control cmds: joint/Cartesian vel/pos + gripper; default `action` = 6× joint vel + gripper |
| rate | 15 Hz (paper / hardware description) |
| cameras | yes (not needed for force residual) |

Paper Appendix B also lists joint **velocities** and EE velocity in the
*recorded* trajectory feature list.

### Missing / insufficient for force residual

| field | status |
|---|---|
| \(\tau_{\mathrm{meas}}\) / effort | **absent** from official RLDS schema and from paper feature list |
| current→torque cal | **n/a** (no torque/current channel in official release docs) |
| \(t\) as wall-clock | **not** in published RLDS `steps` schema (frame/episode structure; no documented per-step timestamp) |
| \(\dot q\) in RLDS obs | **not** listed under official RLDS `observation` (paper claims recorded velocities; RLDS release schema as published does not expose them there) |

**Tier: A0-LIMITED** (state/action / imitation).  
**Not** C0-compatible. Do not infer “Franka + Polymetis ⇒ released torque.”

### Note on third-party remasters

NVIDIA Cosmos3-DROID advertises `joint_torques_computed` /
`motor_torques_measured`. That is a **separate remaster**, not the
official DROID RLDS contract audited here. Until its provenance,
calibration, and timebase are audited on their own, it does **not**
upgrade official DROID to A0-GO.

## AgiBot World

### Present (documented)

| field | status |
|---|---|
| \(t\) | yes — `/timestamp` nanoseconds (real-world releases) |
| \(q\) | yes — `/state/joint/position` |
| \(\dot q\) | yes — `/state/joint/velocity` |
| \(u\) | yes — mainly `/action/joint/position` (+ other cmd channels) |
| joint current | schema key `/state/joint/current_value` |

### Missing / insufficient for force residual

| field | status |
|---|---|
| effort / motor torque | schema keys exist, but docs state **“Effort: … Not available for now.”** |
| wrench | **“Not available for now.”** |
| current→torque calibration | **not published** in the audited README tables |
| force-control capability of the robot | **≠** released calibrated \(\tau\) in the log |

**Tier: A0-LIMITED** (richer proprio + timestamps than DROID RLDS; still
no contract \(\tau_{\mathrm{meas}}\)).

## Bridge / OXE (deferred)

Not first-line for generalized-force residual. Useful later for
representation / cross-embodiment priors only.

## Checklist summary

| question | DROID (official) | AgiBot (docs) |
|---|---|---|
| real \(t\) | unclear / not in RLDS steps schema | yes (ns) |
| joint \(q\) | yes | yes |
| \(\dot q\) measured or reliable | paper yes; RLDS obs schema incomplete | yes |
| \(\tau\) / effort | **no** | keys present, **unavailable** |
| current→torque cal | no | no |
| interpretable \(u\) | yes (cmd streams) | yes (pos cmds) |
| shared timebase state/action | episode steps @ 15 Hz | timestamped h5 |
| invalid/sat flags | not established here | not established here |
| force-auditable? | **no** | **no** |

## Gates

| tier | result |
|---|---|
| A0-GO force-auditable | **fail** (both primaries) |
| A0-LIMITED state-action | **hit** |
| A0-STOP provenance | not needed as aggregate label (schemas are readable; force channel simply absent/unavailable) |

## What may follow

**Allowed (optional, separate ledger):** state/action coverage,
representation shift, crude \(E_v(a)\) proxies from \(q,\dot q\) — never
labeled as physics residual C0.

**Forbidden without new data:** REAL-LOG-A1 force residual;
\(I(\mathcal V;Y\mid a)\) with unknown \(\mathcal V\); renaming any of
this as R10-C0.

## Family implication

\[
\boxed{\textbf{REAL-LOG force-audit line: no A1}}
\]

Physics-validation path without hardware has hit the engineering
boundary stated in the prereg. Prefer: freeze paper conclusions from
R7–R9 + SIM-X; keep R10-C0 locked/deferred; do **not** open a third
simulator family to compensate.
