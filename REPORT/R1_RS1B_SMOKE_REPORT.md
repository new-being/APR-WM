# R1-RS1B Plumbing Smoke Report

Date: 2026-08-15  
Prereg: REPORT/R1_RS1B_PREREG.md  
Artifacts: runs/r1_rs1b/smoke/  
Status: **PLUMBING PASSED; FORMAL CONFIRMATORY RUN NOT STARTED**

## Scope

The preregistered smoke cell was run exactly once:

- seed = 9011
- α = −0.24
- A = 1.5 A₀

It exercises the complete RS1B path while keeping the confirmatory matrix and
GO gates unevaluated:

P_rev → frozen R0.6 candidate → dynamics hard filters → freeze → blind H32.

## Intake

The detector and consequence policy were loaded from the passing, frozen
RS1A.5 artifact; nothing was recalibrated in RS1B.

| Quantity | Value |
|---|---:|
| D₀ | 0.002388 |
| frozen D₀ threshold at 1.5 A₀ | 0.002231 |
| C | 0.003242 |
| frozen C_tol | 0.001934 |
| intake decision | **revise_worthy** |

## Candidate and short-horizon decision

| Quantity | Value |
|---|---:|
| selected operator | abs_v_v |
| α̂ | −0.25464 |
| H2/H4/H8 utility | +0.000521 |
| frozen R0.6 pipeline | **accepted** |

## Dynamics diagnostics and blind H32

| Quantity | Value |
|---|---:|
| minimum effective damping | +0.001341 |
| maximum revision power | −2.0 × 10⁻⁵⁴ |
| positive-power integral | 0 |
| maximum expansion excess | −1.58 × 10⁻⁸ |
| H32 support exit fraction | 0 |
| H32 no-revision RMSE | 0.001717 |
| H32 revised RMSE | 0.0000994 |
| paired H32 gain | **+0.001618** |
| H32 stable | **yes** |

The accept bit is frozen before the blind H32 routine is called. The HDF5
artifact separates raw_truth, learner_visible, decision, diagnostics, and
evaluation; tau_hidden is absent from learner_visible.

## Verification

- Full test suite: **101 passed**
- RS1B and related targeted tests: **14 passed**
- CLI: r1-rs1b registered
- Frozen packages: robosuite 1.5.2, MuJoCo 3.11.0

## Decision

**RS1B plumbing passed; RS1B_GO remains unevaluated.**

The 30-episode held-out matrix must be run without changing
REPORT/R1_RS1B_PREREG.md. Only that formal run may set RS1B_GO and unlock
RS1C.
