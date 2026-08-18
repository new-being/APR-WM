# R10-C0 Acquisition-Chain Feasibility Design

Date: 2026-08-17  
Status: **FROZEN design**; **DEFERRED**; does not unlock C0; does not
choose a vendor  
Depends on: `REPORT/REG/R10/R10_C0_PREREG.md`,
`REPORT/REG/R10/R10_C0_RESIDUAL_H5_SCHEMA.md`  
Does not: APR-WM model; validity; certificate; router; revision;
full-arm C0 as the first attempt

## Question

\[
\boxed{
\text{can real nominal dynamics close to the measurement noise floor?}
}
\]

Not: can we simulate a complex robot.

## A. Platform (do this first)

| platform | rank | why |
|---|---|---|
| 1-DoF rotary hinge / door | preferred | torque + \(q\) easy; attribution simple |
| one arm joint (e.g. Panda single-axis) | next | mature I/O; control stack heavier |
| full arm | later | too many residual sources for C0 |

Preferred plant: one rotary DoF, **no contact** (\(\tau_c\approx 0\)):

\[
\tau_m=I(q)\ddot q+\tau_g(q)+\tau_f(\dot q)+\tau_c+r_{\mathrm{real}}.
\]

C0 attributes failures to sensor / actuator / model, not contact.

Hardware minimum: encoder (or resolver) **and** a torque measurement
chain. \(\tau_{\mathrm{measured}}\succ\tau_{\mathrm{commanded}}\). If
no torque sensor, log \(\tau_{\mathrm{estimated}}\) and the estimator
string (`tau_source`).

## B. Sensor chain

One time axis: \((t,q,\dot q,\ddot q,\tau,u)\).

- \(q\): encoder / resolver / high-res pot; known rate + timestamp.
- \(\dot q\): not a raw low-rate channel by default. Central difference
  or observer. Root attr `qvel_estimator` required.
- \(\ddot q\): noise amplifier. Root attr `qacc_estimator` required
  (Savitzky–Golay / Kalman / observer + parameters). Never results-only.
- \(\tau\): measured preferred; else estimated with documented chain.
- \(u\): executed command, same \(t\), for later \(I(\mathcal V;Y\mid a)\).

## C. Sync

Target \(t_q=t_\tau=t_u\) on a shared controller clock.

If not: store `timebase=offset` and \(\Delta t_{q-\tau}\), \(\Delta t_{q-u}\).
Otherwise \(r=r_{\mathrm{physics}}+r_{\mathrm{timing}}\).

Rate: joint-torque / physics, not 20 Hz policy. `dt` on the file root.

## D. First nominal trajectories

Goal: **noise floor**, not APR-WM.

- No contact, no impact, no saturation, no extreme stick-slip.
- Small multi-sine: \(q(t)=\sum_i a_i\sin(\omega_i t)\), small angle,
  enough \(\dot q,\ddot q\), friction without leaving the smooth regime.
- Many short episodes, not one long rollout: \(N=20\)–\(50\), each
  \(10\)–\(30\,\mathrm{s}\), to estimate \(\mathrm{Var}(r_{\mathrm{real}})\).

Output of this campaign: \(\mathrm{NRMSE}_{\mathrm{real\ floor}}\)
only. Then preregister the C0 threshold. Then `r10-c0`.

## Execution order (frozen)

A platform → B torque chain → C timestamps → D freeze \(qacc\) (and
\(qvel\)) estimator → E contact-free nominal set → F `residual.h5` →
G noise characterization → H preregister \(\mathrm{NRMSE}_{\mathrm{floor}}\)
→ I C0 gate.

Critical decision: **one interpretable real mechanical DoF** — when
hardware exists. Topology frozen as **H1**
(`REPORT/REG/R10/R10_C0_H1_RIG.md`). CAD0 **deferred**.
**Now:** SIM-X (`REPORT/REG/SIMX/SIMX_PREREG.md`), not procurement.
