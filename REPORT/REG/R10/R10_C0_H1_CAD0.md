# R10-C0-H1-CAD0 — Sensing-Plane Inertia Accounting

Date: 2026-08-17  
Status: **FROZEN cut**; **DEFERRED** (no hardware path now);
geometry not drawn; does not unlock C0. Next executable cell is
SIM-X0 (`REPORT/REG/SIMX/SIMX0_PREREG.md`), not this CAD.  
Depends on: `REPORT/REG/R10/R10_C0_H1_RIG.md`,
`REPORT/REG/R10/R10_C0_H1_ENVELOPE.md`  
Does not: estimator; system ID; validity; freeze \(\tau_{\mathrm{FS}}\);
treat paper \(I=0.0533\) as CAD truth

## Measurement cut

H1 chain: motor → coupling → **torque transducer** → hinge/load.

\[
\boxed{
I_{\mathrm{CAD}}
=
\text{equivalent inertia of everything that rotates with }q
\text{ downstream of the torque-sensing plane}
}
\]

C0 remains

\[
\tau_{\mathrm{meas}}=I_{\mathrm{CAD}}\ddot q+\tau_f(\dot q)+r_{\mathrm{real}}.
\]

A CAD \(I\) of the “whole rotating bench” is the wrong number if it
includes the rotor.

## What enters \(I_{\mathrm{CAD}}\)

| part | in \(I_{\mathrm{CAD}}\) | note |
|---|---|---|
| rigid arm / door | yes | main body |
| hinge-side hub | yes | often non-negligible |
| output shaft (if it turns with the load) | yes | |
| transducer **load-side** rotating inertia | yes / datasheet | fill later |
| coupling **load-side** | only if downstream of the sensing plane | upstream: exclude |
| motor rotor | **no** | upstream of transducer |
| motor-side coupling | **no** | same |
| stationary bearing housing | no (inertia) | friction → \(\tau_f\) |
| encoder rotor / codewheel | yes if it turns with the hinge | usually small |

\[
I_{\mathrm{CAD}}=\sum_j I_{j,\mathrm{hinge}},\qquad
I_{j,\mathrm{hinge}}=I_{j,\mathrm{COM}}+m_j d_j^2.
\]

Uniform arm: \(I_{\mathrm{arm}}=\frac13 m L^2\). Solid hub about its
axis: \(\frac12 m_h r_h^2\). Payload: \(m_p d_p^2+I_{p,\mathrm{COM}}\).

Paper \(0.0533\,\mathrm{kg\cdot m^2}\) is **arm-only** slender-rod
\(I_{\mathrm{arm}}\). CAD0 replaces it by the **sum**, not by a more
precise rod formula.

## Friction cut (same plane)

If transducer → shaft bearings → arm, then

\[
\tau_f=\tau_{\mathrm{bearing}}+\tau_{\mathrm{seal}}+\tau_{\mathrm{other,downstream}}.
\]

Motor bearings and cogging are **upstream** of \(\tau_{\mathrm{meas}}\)
and must not be stuffed into load-side nominal residual. That is why
the inline transducer is the C0 instrument.

## CAD0 deliverable (six numbers)

No FEM. Export:

\[
m_{\mathrm{rotating}},\quad
d_{\mathrm{COM}},\quad
I_{\mathrm{CAD}},\quad
q_{\max},\quad
\dot q_{\max},\quad
\ddot q_{\max}.
\]

Then

\[
\tau_{I,\max}=I_{\mathrm{CAD}}\ddot q_{\max},\qquad
\tau_{\mathrm{peak}}=I_{\mathrm{CAD}}\ddot q_{\max}+\tau_{f,\max},
\]

and \(\tau_{\mathrm{FS}}\gtrsim(1.5\sim 2)\tau_{\mathrm{peak}}\).

If CAD later gives e.g. \(I_{\mathrm{CAD}}=0.060\) at
\(\ddot q_{\max}=4\) and \(\tau_{f,\max}=0.10\), then
\(\tau_{\mathrm{peak}}\approx 0.34\,\mathrm{N\cdot m}\) and
\(\tau_{\mathrm{FS,min}}\approx 0.51\sim 0.68\,\mathrm{N\cdot m}\)
(still ~1 Nm class). **Do not freeze FS until \(I_{\mathrm{CAD}}\) exists.**

Next information: **deferred.** Executable next cell is SIM-X0
(`REPORT/REG/SIMX/SIMX0_PREREG.md`), not load-side CAD solids.
