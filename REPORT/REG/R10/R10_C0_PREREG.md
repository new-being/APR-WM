# R10-C0 Preregistration — Real Sensor-Chain C0 Closure

Date: 2026-08-17  
Status: **FROZEN**; **LOCKED** until a real (or hardware-logged) residual
trace exists  
Depends on: `REPORT/REG/R10/R10_PREREG.md`  
Does not: run on the R7–R9 saturated toy; treat robosuite C0 as this
gate; autonomous revision

## Question

Does generalized-force accounting that closed in simulation remain
near the noise floor on the **real sensor chain** under nominal
dynamics?

\[
\boxed{
\text{simulator C0}
\neq
\text{real-C0}.
}
\]

## Lock

No synthetic \(F_{\max}\) substitute. CLI `r10-c0` refuses unless a
preregistered log path is present (hardware or device-logged residual
HDF5). Passing R1-MJ0/RS0 does not unlock this stage.

## After a pass

Unlock **R10-C1** (controlled real mismatch with ground truth). Shadow
commit still off.
