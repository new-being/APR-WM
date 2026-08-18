# R10-C0 Preregistration — Real Sensor-Chain C0 Closure

Date: 2026-08-17  
Status: **FROZEN**; **LOCKED/deferred** (no real mechanical plant;
standard unchanged). SIM-X is a **separate** ledger and is **not**
this gate.  
Depends on: `REPORT/REG/R10/R10_PREREG.md`  
Does not: run on the R7–R9 saturated toy; treat robosuite/MJ0 C0 as
this gate; autonomous revision; C1–C3; copy simulator HDF5 into the
real path

## Question

Does generalized-force accounting that closed in simulation remain
near the **measured** noise floor on the real sensor chain under
nominal dynamics?

\[
r_{\mathrm{real}}=\tau_{\mathrm{measured}}-\tau_{\mathrm{nominal}}.
\]

\[
\boxed{\text{simulator C0}\neq\text{real-C0}.}
\]

## Lock

Only `runs/r10_c0/real/residual.h5` with
`schema=aprwm.r10_c0.residual.v1` and
`source\in\{\mathrm{hardware},\mathrm{device\_log}\}`.
`r10-c0` rejects if missing. MJ0/RS0/MuJoCo truth dumps are invalid
even if renamed into that path.

Schema: `REPORT/REG/R10/R10_C0_RESIDUAL_H5_SCHEMA.md`.  
Acquisition design (platform / sync / first trajectories):
`REPORT/REG/R10/R10_C0_ACQUISITION_DESIGN.md`.  
H1 topology: `REPORT/REG/R10/R10_C0_H1_RIG.md`.  
Envelope worksheet: `REPORT/REG/R10/R10_C0_H1_ENVELOPE.md`.  
CAD0 sensing-plane cut: `REPORT/REG/R10/R10_C0_H1_CAD0.md`
(**deferred**; next work is SIM-X0).

NRMSE threshold is **not** \(10^{-4}\). Preregister
\(\mathrm{NRMSE}_{\mathrm{floor}}\) after the first hardware noise
campaign.

## Platform (collection, not a gate)

Prefer a **single-DoF hinge / door** with measurable torque, synced
joint state, software safety limits, and a device log — before a full
arm chain — so C0 failures can be attributed to sensor / actuator /
model.

Timebase: \(t_q=t_\tau=t_u\), or an explicit \(\Delta t\) in the file
root. Otherwise residual mixes timing error with physics mismatch.

## After a pass

Unlock **R10-C1** only. C1–C3 stay locked until C0. Shadow commit off.
