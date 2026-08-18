# SIM-X0 MuJoCo ↔ APR-WM force-space accounting

Date: 2026-08-17  
Status: **FROZEN** for SIM-X0; inherited by SIM-X1+ unless this cell fails  
Depends on: `aprwm_v0/mujoco_force.py` (`generalized_force_residual`)  
Does not: R10-C0; hardware \(\tau_{\mathrm{meas}}\); truth `qfrc_passive`
as a known term; detector / validity / certificate

## Claim this file is allowed to support

Convention alignment:

\[
\boxed{
\text{MuJoCo dynamics convention}
\;\leftrightarrow\;
\text{APR-WM force-space convention}
}
\]

Not: real-physics validation. Not: smaller residual is better science.

## Identity (learner residual)

Same generalized-force residual as R0.1 / R1-MJ0, on the **SIM-X0
host plant**:

\[
r
=
M(q)\ddot q
+
c(q,\dot q)
-
\tau_{\mathrm{known}}
-
\tau_{\mathrm{constraint}}
-
\tau_{\mathrm{nominal\text{-}passive}}
\]

MuJoCo buffers (same physics step \(k\)):

| symbol | buffer / rule |
|---|---|
| \(q,\dot q,\ddot q\) | residual-row \(q,\dot q\) = **pre-`mj_step`** oracle state; \(\ddot q\) = `qacc` of that step |
| \(M(q)\) | `mj_fullM` / CSR `data.M` — **never** deleted `qM` |
| \(c(q,\dot q)\) | `qfrc_bias` |
| \(\tau_{\mathrm{known}}\) | `qfrc_actuator + qfrc_applied` |
| \(\tau_{\mathrm{constraint}}\) | `qfrc_constraint` (X0: \(\approx 0\); no contact, no limit hit) |
| \(\tau_{\mathrm{nominal\text{-}passive}}\) | **declared** XML damping (and frictionloss if XML has it) via `NominalPassiveParams`, evaluated at the **pre-`mj_step`** \(\dot q\) |

After `mj_step`, `qacc` / `qfrc_*` still belong to the pre-step force
evaluation, while `qpos`/`qvel` have advanced. Logging the residual
row's \(q,\dot q\) as that pre-step state keeps the identity aligned.
Using post-step \(\dot q\) for `NominalPassiveParams` is a **timing /
passive** interface bug (hidden when damping \(=0\), as in R1-MJ0).

\[
\boxed{
\text{truth } \texttt{qfrc\_passive}
\;\notin\;
\text{learner residual}
}
\]

Log it under `raw_truth` / `diagnostics` only. Feeding it in is
**label leakage**, not closure.

On X0, \(\tau_{\mathrm{known}}\) is the oracle applied/generalized
force (prefer `qfrc_applied` on the hinge; actuators only if they are
1:1 with that DoF and **unsaturated**).

## Sign / bias / passive / constraint (failure atlas)

If \(r\) does not sit on the numerical floor, **stop**. Order:

1. **Sign:** \(+c\) vs \(-c\); \(\tau_{\mathrm{known}}\) subtracted
   vs added; hinge axis left-handed.
2. **Bias:** gravity torque on a non-vertical axis; constant
   `qfrc_applied` offset; unit (N·m vs N).
3. **Passive:** learner damping \(\neq\) `dof_damping`; using
   `qfrc_passive` truth; `frictionloss` in XML but not in nominal
   (MuJoCo puts `frictionloss` in `qfrc_constraint`, not
   `qfrc_passive`); **post-step \(\dot q\)** fed to
   `NominalPassiveParams` while `qacc`/`qfrc_*` are pre-step.
4. **Constraint:** joint limits, contacts, welds. X0 forbids these
   as physics; a large `qfrc_constraint` is an **interface/plant**
   fail.

Do **not** increase model capacity, train a detector, or start X2.

## Floor (not a beauty contest)

X0 asks: is \(r\) at the **integrator / float** floor of this plant,
not whether NRMSE can be driven to \(10^{-16}\).

Frozen gate number (justified by the same 1-DoF no-contact class as
R1-MJ0, which closed at \(\sim 10^{-11}\) against a \(10^{-4}\) bar):

\[
\mathrm{NRMSE}_{\tau} < 10^{-4}.
\]

A pass at \(10^{-5}\) is not “more real” than a pass at \(10^{-8}\).
A fail at \(10^{-2}\) with a structured bias is an adapter fail.

## What this is not

Not R1-MJ0 **credit**. MJ0 closed a *different program cell* (R1
realism bridge). SIM-X0 must close on the **plant that will host
SIM-X1**. Reusing `generalized_force_residual` is required;
copying `runs/r1_mj0/` into SIM-X0 is **not** a pass.

Not R10-C0. Artifacts: `runs/sim_x0/…`, `source=simulator` /
`sim-x0`. Never `runs/r10_c0/real/`.
