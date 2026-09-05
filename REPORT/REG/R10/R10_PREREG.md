# R10 Preregistration — Real-Physics Shadow Validation

Date: 2026-08-17  
Status: **FROZEN** (family); first gate **R10-C0 = locked/deferred**
(no hardware); **SIM-X = STOP after X3**; **REAL-LOG-A0 = A0-LIMITED**
(separate ledger; not this family)  
Depends on: `REPORT/REP/R9/R9_TOY_FAMILY_STOP.md`  
Does not: autonomous revision; certificate-commanded high-risk motion;
R9-C0; adaptive B1; new toy \(F_{\max}\) plant; salvage B1-P0 NetVoI;
treat MuJoCo / PhysX / Isaac as this family; lower C0 because SIM-X
or REAL-LOG exists; treat public demos as residual.h5

**Physics (two layers).** Explicit physics = structured terms *in the
model*. R10 real physics = closed-loop dynamics of a **real
mechanical plant** plus contracted sensors. No such plant now
\(\Rightarrow\) **R10-C0 cannot be run.** Do not relabel simulators.

SIM-X stop: `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`.  
REAL-LOG-A0: `REPORT/REP/REALLOG/REALLOG_A0_REPORT.md` — public logs are
**state-action limited**, not force-auditable; still never R10-C0.

## Why R10 still exists

R9’s remaining questions *eventually* depend on the **operating
distribution of a real system**, not on another algorithm cell of the
saturated toy:

- Do task actions naturally excite validity-relevant modes?
- Does a conservative controller induce epistemic blindness?
- Which faults (saturation, friction, backlash, payload) are
  self-exciting under normal operation?
- How long does one real calibration last?
- What is the time / wear / opportunity cost of a dedicated re-probe?

Continuing to design those quantities on the toy plant risks fitting an
environment to an epistemic policy.

## Mode

APR-WM runs **online in shadow**:

\[
\mathrm{calibrate}\to b(\mathcal V)\to L_{\mathrm{WM}}\to
\text{shadow commit.}
\]

Log useful / harmful **counterfactuals**. The bit does not drive the
robot. Full closed-loop revision waits until this lifecycle holds on
hardware.

## Four checks (in order)

1. **R10-C0 — Real-C0 closure**  
   On the real sensor chain, does generalized-force residual stay near
   the noise floor in nominal dynamics? Simulator C0 (R1-MJ0/RS0) is
   not a substitute.

2. **R10-C1 — Controlled real mismatch**  
   Inject a **ground-truth** change (software actuator limit, added
   payload, damping, or controllable friction). Not “unknown real
   world” yet.

3. **R10-C2 — Action-conditioned validity observability**  
   Core question: does \(I(\mathcal V;Y\mid a)\) change materially
   with the executed action? Same real shift, different \(a\).
   Which motions expose validity, and which induce
   policy-induced epistemic blindness?

4. **R10-C3 — Shadow certificate lifecycle**  
   Replay / live-shadow the R9 objects (frozen where they still apply)
   and report would-be useful / harmful commits. Still no actuation
   authority.

C1–C3 stay **locked pending C0**. Hardware path (when a plant exists)
is H1 (`REPORT/REG/R10/R10_C0_H1_RIG.md`). Inertia for
\(\tau_{\mathrm{nominal}}\) is CAD0 sensing-plane \(I_{\mathrm{CAD}}\)
(`REPORT/REG/R10/R10_C0_H1_CAD0.md`). **Do not start CAD now.**

SIM-X0–X3 are **not** R10-C0–C3. Passing SIM-X does not unlock C0.

## After C3

Choose, from **data**, whether the surviving need is passive monitoring,
periodic calibration, self-validating actions, or explicit active
probing. Do not decide that on the toy plant.
