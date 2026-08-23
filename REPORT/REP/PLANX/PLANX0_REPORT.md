# PLAN-X0 Report — Teacher / Sensitivity Instrument

Date: 2026-08-18  
Status: **`plan_x0_passed=true`**; **PLAN-X1 unlocked**  
Prereg: `REPORT/REG/PLANX/PLANX0_PREREG.md`  
Artifacts: `runs/plan_x0/formal/`  
Does not: train a proposal net; claim \(AUC_S\) / \(R_B\); touch CAP-X2;
unlock R10

## Question

Is the reachable-target + teacher \(A^\star\) + FD \(h\) pipeline a
closed, non-degenerate instrument?

## Result

\[
\boxed{\texttt{plan\_x0\_passed=true}}
\]

All four instrument gates PASS. Sensitivity is **not** degenerate
(\(Q_{0.9}/Q_{0.1}\approx 102\ge 5\)). PLAN-X1 may be frozen next.

## Implementation header (locked before \(h\) histograms)

| item | value |
|---|---|
| \(U_{\max}\) | 1.5 |
| \(\epsilon_h\) | 0.05 |
| teacher | L-BFGS-B, \(\mathrm{maxiter}=8\) |
| \(A\) | \(\mathbb{R}^{60}\) (20 knots × 3) |

## Data

| split | scenes | conditions | \(S_{\mathrm{feasible}}\) | teacher \(J^\star\le J_{\mathrm{gen}}\) | mean \(J_{\mathrm{gen}}\) | mean \(J^\star\) |
|---|---:|---:|---:|---:|---:|---:|
| train | 128 | 4096 | 1 | 1 | 1008 | 428 |
| val | 32 | 1024 | 1 | 1 | 1014 | 432 |
| test | 64 | 2048 | 1 | 1 | 952 | 393 |

Teacher reduces cost by \(\sim 2.3\times\) vs the generating sequence
(warm start is not left as the expert).

## Gates

| gate | result | detail |
|---|---|---|
| **G0** | PASS | \(S_{\mathrm{feasible}}=1\) (construction; \(\varepsilon_q\) reconstruction of \(A_{\mathrm{gen}}\)) |
| **G1** | PASS | teacher_frac \(=1\ge 0.99\) |
| **G2** | PASS | \(P(h_j>0)=0.986\ge 0.90\) |
| **G3** | PASS | \(Q_{0.9}(h^+)/Q_{0.1}(h^+)=102\ge 5\) |
| **G-label** | PASS | `runs/plan_x0/`; no proposal claim |

## Claims ceiling

**Allowed:** instrument is closed enough to freeze PLAN-X1 (H1/H2
Gaussian proposal; trace-matched covariance).

**Forbidden:** sampling efficiency; B2 vs B1; beating CEM; “sensitivity
improves planning.”

## Unlock

\[
\boxed{\text{PLAN-X0 PASS}\;\Rightarrow\;\text{freeze PLANX1\_PREREG.md}}
\]
