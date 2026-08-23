# PLAN-X1 Report — Unimodal Sensitivity-Aware Gaussian Proposal

Date: 2026-08-19  
Status: **`plan_x1_passed=false`**; pattern **`anisotropy_no_value`**  
Prereg: `REPORT/REG/PLANX/PLANX1_PREREG.md` (gates **not** retuned)  
Data: `runs/plan_x0/formal/` (2048 test conditions)  
Artifacts: `runs/plan_x1/formal/`  
Does not: diffusion; retune \(\sigma_0\)/\(\lambda_h\); reopen CAP-X2-P0;
unlock R10; treat X0 \(h\)-diversity as H2 confirmation

## Question

At equal \(\mathrm{tr}(\Sigma)=60\sigma_0^2\), does
\(\Sigma_{\mathrm{sens}}\) beat isotropic \(\sigma_0^2 I\) around the
**same** learned \(\mu_\phi\)?

## Pattern (pre-declared)

\[
\boxed{\texttt{anisotropy\_no\_value}}
\]

Even **oracle** finite-difference \(h\) (B3) does not beat isotropic
exploration around the same mean. H2 is **negated** on this host/task,
not merely “\(\hat h\) unlearned.” Per prereg: **STOP H2** — do not
rescue with a bigger sensitivity head.

H1 is a **separate** positive: learned \(\mu\) beats a zero/broad mean.

## Layer 1 — H1 (where to search)

| contrast | result |
|---|---|
| \(J(\mu_\phi)\) vs \(J(0)\) | \(450\) vs \(755\); \(\Delta=305\), 95% CI \((291,319)>0\) → **G0 PASS** |
| Spearman \((\log\hat h,\log h)\) | \(0.875\), CI \((0.872,0.878)>0\) → **G1 PASS** |
| \(AUC_S\) B1 − B0 | \(+0.080\), CI \((0.062,0.099)>0\) |

Learned attractor location is real. \(\hat h_\phi\) also tracks \(h\)
well — so G3 failure is **not** “the net never saw curvature.”

## Layer 2 — H2 (how wide per direction) **core, fails**

Same \(\mu\), same trace, same \(B\in\{16,32,64,128,256\}\):

| sampler | \(AUC_S\) |
|---|---:|
| B0 broad prior | 0.169 |
| **B1** \(\mu+\sigma_0^2 I\) | **0.249** |
| B2 \(\mu+\Sigma_{\mathrm{sens},\phi}\) | 0.149 |
| B3 \(\mu+\) oracle \(\Sigma_h\) | 0.130 |

\[
AUC_S^{B2}-AUC_S^{B1}=-0.100,\quad
\text{95\% CI }(-0.113,-0.088)<0
\]

**G3 FAIL.** Sensitivity-shaped covariance is **worse** than isotropic
around the same mean.

Success vs budget \(S(B)\):

| \(B\) | B0 | B1 | B2 | B3 |
|---:|---:|---:|---:|---:|
| 16 | 0.042 | **0.150** | 0.104 | 0.095 |
| 32 | 0.081 | **0.197** | 0.130 | 0.113 |
| 64 | 0.143 | **0.253** | 0.148 | 0.128 |
| 128 | 0.243 | **0.299** | 0.171 | 0.148 |
| 256 | **0.379** | 0.346 | 0.190 | 0.162 |

## Layer 3 — B3 diagnostic

\[
AUC_S^{B3}-AUC_S^{B1}=-0.120,\quad
\text{CI }(-0.132,-0.108)<0
\]

B3 is **not** better than B1. Therefore:

\[
\boxed{
\text{even with known }h,\text{ allocating }\sigma_j^2\propto 1/h_j
\text{ does not improve (and here hurts) search.}
}
\]

This is **not** `sensitivity_prediction_failure` (that required B3>B1
and B2\(\approx\)B1).

Equal-cost geometry **G2 FAIL**: \(CV_{\mathrm{sens}}=0.995\not\le
0.7\times CV_{\mathrm{fixed}}=0.874\). Predicted 1\(\sigma\) ellipsoid
is not close enough to equal-\(\Delta J\).

## G4 — proposal-seeded CEM (passes, H1 not H2)

Vanilla CEM vs first-iter B2 proposal, equal \(B_{\mathrm{total}}\):

| \(B_{\mathrm{tot}}\) | vanilla \(S\) | proposal-CEM \(S\) |
|---:|---:|---:|
| 128 | 0.025 | 0.058 |
| 256 | 0.026 | 0.070 |
| 512 | 0.020 | 0.072 |
| 1024 | 0.025 | 0.072 |

\(\Delta AUC_S=+0.045\), CI \((0.036,0.054)>0\) → **G4 PASS**.

Read as: **learned \(\mu\) as CEM init helps** vs a broad vanilla start.
It does **not** rehabilitate H2: B2/B3 sampling still lose to B1.

Absolute CEM success remains low (\(\approx0.07\)) under the frozen
budget — consistent with CAP-X2-P0 (oracle CEM \(S=0.708\) only at much
larger search, and that was a different success construction).

## Gates

| gate | result |
|---|---|
| G0 mean | **PASS** |
| G1 Spearman | **PASS** |
| G2 equal-cost CV | FAIL |
| G3 \(AUC_S\) B2>B1 | FAIL (core) |
| G4 proposal-CEM | **PASS** (H1 seeding) |
| G-label | PASS |

`plan_x1_passed=false`.

## Claims ceiling

**Allowed:** on `capx_arm3.v1` reachable-target oracle planning, a
learned mean replaces a zero action as a useful basin center; diagonal
\(H_J^{-1}\) variance allocation does not buy sampling efficiency vs
isotropic \(\sigma_0\) at matched trace; X0’s huge \(h\)-anisotropy is
real but **not sufficient** for H2.

**Forbidden:** “sensitivity-aware proposal success”; diffusion as a
silent rescue of H2; mixing into CAP \(R_P\); R10.

## Split (the scientifically useful part)

The original idea was two propositions. Only one is rejected.

\[
\boxed{\textbf{H1 SUPPORTED:}
\text{ experience }\to\text{ action basin center }\mu_\phi(c)}
\]

\[
\boxed{\textbf{H2 REJECTED on this host/task:}
\text{ physics sensitivity }\sigma_j^2\propto 1/h_j
\text{ does not improve search}}
\]

G1 (Spearman \(0.875\)) plus B3 \(<\) B1 shows the covariance **design**
failed, not the curvature predictor. Local Hessian \(\approx\) search
optimal \(q(A)\) was the bad identification: CEM needs mass on
\(\Omega_{\mathrm{success}}\), not a quadratic fit at \(A^\star\).
Trace-matched thin ellipsoids can reduce hit probability vs an
isotropic ball when the success set is not aligned with \(H_J\) axes.

**Not rejected:** learning an action attractor / using \(\mu_\phi\) as
CEM warm start (G4). **Rejected:** letting physics curvature **shape**
the action-search distribution.

\[
\boxed{
\text{experience }\to\text{ action attractor distribution};
\quad
\text{physics }\to\text{ reject/improve candidates}
}
\]

— not \(\text{physics}\to\text{shape }q(A)\). That is the
world-model / action-prior split, not a route failure.

## Unlock / stop

\[
\texttt{anisotropy\_no\_value}
\;\Rightarrow\;
\textbf{STOP H2 on this instrument}
\]

Do not enlarge the sensitivity head or retune \(\sigma_0\) to flip G3.
**PLAN-X2 is not triggered** (`proposal_failure` did not occur).
CAP-X3 stays locked.

PLAN-X1.5 is closed: **FAIL / `iso_sufficient`**
(`REPORT/REP/PLANX/PLANX15_REPORT.md`). Mix-4 does not beat mean+iso
on the frozen \(AUC_S\) gate. Diffusion is **not** unlocked by that FAIL.
