# REAL-LOG-A0 Preregistration — Public Real-Robot Log Feasibility Audit

Date: 2026-08-17  
Status: **FROZEN**; executable; **not R10**; does not unlock R10-C0  
Depends on: `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`  
Does not: train models; download full corpora; compute
\(r=\tau_{\mathrm{meas}}-\tau_{\mathrm{nominal}}\); call logs “real
physics validation”; open a third simulator family; dwell rescue of
SIM-X3

## Why now

SIM-X answered the cheap sim questions: action-conditioned observability
holds cross-engine; persistent lifecycle blindness does not. Marginal
value of another simulator is low. Without hardware, the highest-gain
cheap cell is:

\[
\boxed{
\text{Do any public real-robot logs have enough temporal/dynamics
fields for a meaningful offline real-log shadow audit?}
}
\]

Not: can they fake R10-C0? **No.**

## Question

Schema / provenance audit only. No algorithms.

Required checklist per dataset:

\[
\{t,\ q,\ \dot q,\ \tau/\mathrm{effort/current},\ u\}
\]

1. Is \(t\) a real timestamp or frame index?
2. Is \(q\) raw joint-space state?
3. Is \(\dot q\) measured, or reconstructible from high-rate \(q\)?
4. Is there joint torque / effort / motor current?
5. If only current: is current→torque calibration published?
6. Is \(u\) joint torque, joint position, Cartesian delta, or policy cmd?
7. Do state/action share an interpretable timebase?
8. Are there controller / saturation / invalid flags?
9. Are there continuous same-embodiment trajectories for
   action-conditioned residual **proxy** studies?

## Candidates (order)

1. **DROID** (first)
2. **AgiBot World** (second)
3. BridgeData V2 / OXE — later; not first-line for force residual

## Tiers (only these)

| tier | meaning | next |
|---|---|---|
| **A0-GO: force-auditable** | \(t,q,\dot q\) (or reliable), \(\tau_{\mathrm{meas/effort}}\), \(u\) with documented units/timebase | may open **REAL-LOG-A1** offline residual characterization (**still not R10-C0**) |
| **A0-LIMITED: state-action only** | \(t,q,u\) (maybe \(\dot q\)); no trustworthy torque/effort | representation / \(E_v(a)\) / action prior only; **no** contract residual |
| **A0-STOP: provenance insufficient** | broken timebase, unclear effort, untraceable resampling, no calibration | REAL-LOG family STOP |

## Forbidden

Inventing \(\tau\) from current without calibration. Using third-party
remasters as if they were the official release without a separate
provenance note. Training to “fill” missing fields.

## After A0

If no force-auditable source: physics-validation mainline hits

\[
\boxed{\textbf{missing data source, not missing algorithm}}
\]

R10-C0 stays locked until hardware (or a future force-auditable release).
