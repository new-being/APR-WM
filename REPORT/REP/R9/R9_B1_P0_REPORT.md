# R9-B1-P0 Report — Periodic Active Revalidation Feasibility

Date: 2026-08-17  
Status: **`R9_B1_P0_PASS=false`**; pattern
**dwell_bounded_uneconomic_on_shift**; locus **shift_value**;
adaptive R9-B1 **locked**  
Prereg: `REPORT/REG/R9/R9_B1_P0_PREREG.md`  
Artifacts: `runs/r9_b1_p0/formal/summary.json`

## Question

With licensed cons epistemically blind, does re-running the **frozen**
P0/A0 calibration every \(M=K_{\min}=2\) tasks bound stale dwell
without a detector, at acceptable NetVoI?

No new model. No \(M\)-sweep. Phases: `at_probe` (\(k=3\)) and
`after_probe` (\(k=2\)).

## Gates

| Gate | result |
|---|---|
| G-revalidation stay / benign / invalid | **✓ \(1.0/1.0/1.0\)** |
| G-stale \(K_{\mathrm{stale}}\le 1\), repeat \(=0\) | **✓** (max stale \(1\); first harm \(3\); repeat \(0\)) |
| G-benign \(2.0\to 2.2\) still licensed | **✓** |
| G-stable-cost stay \(\mathrm{NetVoI}>0\) | **✓ \(2.37\times10^{-3}\)** |
| G-shift-value invalid `after_probe` periodic \(>\) never | **×** \(-5.88\times10^{-3} < -0.97\times10^{-3}\) |

\[
\boxed{\texttt{R9\_B1\_P0\_PASS}=\text{false}}
\]

Invalid harmful commits, worst phase: never-reprobe \(15\) \(\to\)
periodic \(3\). Repeated harm is gone. A0’s logistic still classifies
the \(u=1\) probe correctly in the lifecycle.

Stay blocks can pay a probe every two tasks. Invalid-shift blocks
cannot: two extra \(C_{\mathrm{cal}}\) plus post-revoke abstention
(including \(\alpha\) that still have \(A>0\) at \(F_{\max}=1.5\))
outweigh the avoided harm in \(J\)-units.

## Reading

Periodic revalidation **does** what B0’s CUSUM could not: it restores
the \(u=1\) excitation on a finite certificate age, so
\(K_{\mathrm{stale}}\le 1\) and \(N_{\mathrm{harmful,repeat}}=0\).

It does **not** beat never-reprobe on invalid-shift NetVoI. Safety
dwell is bounded; the billed surveillance is not worth this task
margin once the plant is already invalid.

Do not sweep \(M\). Adaptive B1 stays locked. R9 toy-family **STOP**:
`REPORT/REP/R9/R9_TOY_FAMILY_STOP.md`. Next is R10 shadow real-C0,
not a constrained-control rewrite of this NetVoI gate.
