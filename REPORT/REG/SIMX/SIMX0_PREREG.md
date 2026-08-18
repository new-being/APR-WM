# SIM-X0 Preregistration — Nominal Force Accounting Closure

Date: 2026-08-17  
Status: **FROZEN**; **PASS** (`REPORT/REP/SIMX/SIMX0_REPORT.md`);
unlocks SIM-X1 only; **not an R10 gate**  
Depends on: `REPORT/REG/SIMX/SIMX_PREREG.md`,
`REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md`  
Does not: unlock R10-C0; SIM-X1–X3; detector; validity; certificate;
\(I(\mathcal V;Y\mid a)\); RoboCasa; H1 CAD; copy `runs/r1_mj0` as a
pass; treat small residual as real physics

## One question

\[
\boxed{
\text{On a minimal MuJoCo articulation, does frozen nominal
force accounting put }r\text{ on the numerical/integrator floor?}
}
\]

That is **convention alignment**, not a full APR-WM rerun.

## Plant (minimal, frozen)

- 1-DoF **revolute**
- **no** contact (no colliding geoms; no weld that loads the hinge)
- **no** payload shift
- **no** actuator saturation (`ctrlrange` / clamps off)
- nominal damping/friction **known** (XML = `NominalPassiveParams`)
- hinge axis **vertical** so \(\tau_g\simeq 0\) (or gravity off)
- joint **unlimited**, or motion strictly interior — no limit
  constraint
- engine = native MuJoCo (raw `mujoco` or robosuite **only as host**);
  not SAPIEN; not the R7–R9 Python ODE relabeled as MuJoCo

Oracle, same step \(k\): \(q,\dot q,\ddot q\) and applied/generalized
force from `mjData`. **No** \(\ddot q\) estimator.

## Residual

`REPORT/REG/SIMX/SIMX0_FORCE_ACCOUNTING.md`. Must call
`aprwm_v0.mujoco_force.generalized_force_residual`. Must **not**
subtract truth `qfrc_passive`.

## Forbidden in this cell

No detector. No validity. No certificate. No action-conditioned
comparison. No WM / NetVoI. No CUSUM. No payload / \(F_{\max}\) /
friction **shift** (those are X1).

## Gates

| id | pass iff |
|---|---|
| **G-min** | plant matches the list above |
| **G-oracle** | \(q,\dot q,\ddot q,\tau_{\mathrm{known}}\) are `mjData` at \(k\), not filtered policy-rate |
| **G-leak** | learner residual excludes truth `qfrc_passive` |
| **G-close** | every frozen excitation cell has \(\mathrm{NRMSE}_{\tau}<10^{-4}\) |
| **G-engine** | MuJoCo, not PhysX |
| **G-label** | `runs/sim_x0/`; never `source=hardware`; never R10 path |

Excitation: a small frozen set (e.g. sine / chirp / piecewise hinge
torque), duration and `dt` declared in the report. Not a task policy.

## Fail ⇒ interface, not capacity

If G-close fails, diagnosis order in the accounting note
(sign → bias → passive → constraint). Then stop. Do not open X1.

## Pass ⇒ unlock SIM-X1 only

X1 = oracle mismatch on **this same plant**. X2
(\(I(\mathcal V;Y\mid a)\)) stays locked until X1, so an X2
information gap cannot be an **adapter error**.

R10-C0 stays locked. `real physics` is not claimed.
