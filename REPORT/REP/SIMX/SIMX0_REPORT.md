# SIM-X0 Report — Nominal Force Accounting Closure

Date: 2026-08-17  
Status: **`sim_x0_passed=true`**; unlocks **SIM-X1 only**  
Does not: unlock R10-C0; claim real physics; run validity / certificate /
\(I(\mathcal V;Y\mid a)\)

## Question

On the frozen SIM-X host plant (`simx_hinge.v1`), does nominal
generalized-force accounting put \(r\) on the numerical floor?

\[
\mathrm{SIM\text{-}X0\ PASS}
\iff
\text{same-host-plant nominal force accounting closes below }10^{-4}.
\]

## Plant

| item | value |
|---|---|
| host | `simx_hinge.v1` (`aprwm_v0/simx_plant.py`) |
| DoF | 1 revolute, vertical axis |
| drive | `qfrc_applied` only (`nu=0`) |
| contact | `contype=conaffinity=0` |
| limits | unlimited |
| damping | XML \(=\) `NominalPassiveParams` \(=0.10\) |
| frictionloss | \(0\) |
| engine | MuJoCo 3.11.0 |
| xml_sha256 | `5409226b203f4eb891bb941d21858c45e9ae65d9bbf9f4b1dc8bc310c9bd02ef` |

This plant is the **X1 host**. X0 is not a disposable smoke body.

## Matrix

Seeds \(\{9101,9111,9121\}\) × trajectories
\(\{\mathrm{sine},\mathrm{chirp},\mathrm{piecewise}\}\) = 9 cells.
\(T=10\,\mathrm{s}\), \(\mathrm{d}t=0.002\).

## Result

| gate | value |
|---|---|
| max \(\mathrm{NRMSE}_{\tau}\) | \(9.56\times 10^{-17}\) |
| G-close (\(<10^{-4}\)) | pass |
| max \(\lvert qfrc\_constraint\rvert\) | \(0\) |
| max \(\lvert qfrc\_actuator\rvert\) | \(0\) |
| `unlocks_r10_c0` | **false** |
| `unlocks_sim_x1` | **true** |
| `real_physics` | **false** |

Artifacts: `runs/sim_x0/formal/` (`summary.json`, per-cell `.h5`).

## Interface finding (atlas: passive / timing)

First short run with post-step \(\dot q\) in `NominalPassiveParams`
failed at \(\mathrm{NRMSE}\sim 1.8\times 10^{-3}\) (atlas hit passive).
Cause: after `mj_step`, `qacc`/`qfrc_*` belong to the **pre-step**
force evaluation, while `data.qvel` has advanced. R1-MJ0 hid this by
using damping \(=0\).

Fix (frozen in `SIMX0_FORCE_ACCOUNTING.md` and
`generalized_force_residual(..., qvel_for_passive=)`): evaluate
declared damping at **pre-`mj_step`** \(\dot q\); log that
\((q,\dot q)\) on the residual row. No network, no detector.

## Claims

Allowed: MuJoCo \(\leftrightarrow\) APR-WM force-space convention closes
on `simx_hinge.v1`.

Not allowed: real-physics validation; R10-C0; X2 information claims.

## Next

Define **SIM-X1** oracle mismatch family on **this same** host plant.
X2–X3 stay locked. R10-C0 stays locked/deferred.
