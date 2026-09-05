# R1-RS1C Plumbing Smoke Report

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS1C_PREREG.md`  
Artifacts: `runs/r1_rs1c/smoke/`  
Status: **PLUMBING PASSED.** Confirmatory 100-episode matrix and `RS1C_GO` are **not** evaluated here.

This is a new policy hypothesis. It does **not** rewrite `RS1B_GO`.

## Scope

Preregistered smoke cells, seed `9041`:

| α | \(A/A_0\) | Role |
|--:|----------:|------|
| 0 | 0.5 | tolerate-scale |
| 0 | 1.5 | tolerate-scale |
| −0.24 | 0.5 | probe → VoI \(A^\star\) |
| −0.24 | 1.5 | revise-worthy (direct) |

H32 uses the RS1B intervention sampler, not RS1B.2 targeted bands.
Support never enters the accept bit.

## Frozen intake

Loaded from passing `runs/r1_rs1a5/formal/summary.json`:
\(C_{\mathrm{tol}}=0.001934\), \(A^\star=1.5A_0\), \(\lambda=0.0015\).
Per-amplitude \(D_0\) thresholds were not recalibrated.

## Cells

| Cell | \(C\) | bucket \(i\to f\) | install | monitor | H32 gain |
|------|------:|-------------------|:-------:|---------|---------:|
| \(\alpha=0\), \(0.5A_0\) | \(\approx 0\) | tolerate → tolerate | no | none | — |
| \(\alpha=0\), \(1.5A_0\) | \(\approx 0\) | tolerate → tolerate | no | none | — |
| \(\alpha=-0.24\), \(0.5A_0\) | 0.00430 | probe → revise_worthy (VoI flip) | **yes** | normal (\(I_{\mathrm{exit}}=0\)) | **+0.00370** |
| \(\alpha=-0.24\), \(1.5A_0\) | 0.00430 | revise_worthy → revise_worthy | **yes** | normal (\(I_{\mathrm{exit}}=0\)) | **+0.00370** |

VoI cell: extra \(1.5A_0\) evidence, \(V=+0.000922>0\), promoted, then passivity + short utility, then freeze, then blind H32. `accepted_flipped_by_support=false`. Operator `abs_v_v`. No passivity violation.

The two installed cells share \(A^\star\) evidence, so the candidate/H32 numbers match. That is expected, not a leak of the accept bit.

## Verification

- Targeted tests: **18 passed** (`test_r1_rs1c` + RS1B/B.1/B.2)
- CLI: `r1-rs1c --smoke`
- Frozen packages: `robosuite==1.5.2`, `mujoco==3.11.0`

## Decision

**RS1C plumbing passed; `RS1C_GO` remains unevaluated.**  
`RS1B_GO` unchanged (`false`). RS2 still locked. Next step is the held-out 100-episode formal matrix without changing gates.
