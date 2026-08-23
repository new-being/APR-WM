# PLAN-X0 Preregistration — Proposal Benchmark / Teacher / Sensitivity Instrument

Date: 2026-08-18  
Status: **FROZEN**; **formal PASS** (`REPORT/REP/PLANX/PLANX0_REPORT.md`);
PLAN-X1 prereg frozen (`REPORT/REG/PLANX/PLANX1_PREREG.md`)  
Depends on: `REPORT/REG/PLANX/PLANX_PREREG.md`, host `capx_arm3.v1`  
Does not: train \(\mu_\phi/\hat h_\phi\); compare B0–B3; CEM vs
proposal-CEM; diffusion; \(\rho>0\) residual plant; unlock R10;
retune CAP-X2-P0; claim \(AUC_S\) or \(R_B\)

## One question

\[
\boxed{
\text{Are reachable-target conditions, offline teacher }A^\star,
\text{and finite-difference curvatures }h\text{ a closed, non-degenerate
instrument for later sensitivity-aware proposals?}
}
\]

X0 **must not** claim that experience or sensitivity improves planning.

## Plant / action (frozen)

\[
\boxed{\texttt{capx\_arm3.v1}}
\]

| item | frozen |
|---|---|
| DoF | 3 revolute; no contact |
| dynamics | **oracle** (true plant; no learned WM) |
| horizon | \(1.0\,\mathrm{s}\) |
| model \(\Delta t\) | \(0.01\,\mathrm{s}\) |
| action knots | 20 piecewise-constant |
| \(A\) | \(\mathbb{R}^{60}\) |
| \(\mu\) Coulomb | 0 (schema only; same CAP-X freeze) |

Condition (learner-visible at test; X0 stores oracle fields separately):

\[
\boxed{c=(q_0,\dot q_0,q^\star,\theta_e)}
\]

Hidden at test: generating sequence \(A_{\mathrm{gen}}\) and teacher
internals.

## Reachable-target construction (frozen)

For each condition:

1. Sample a legal smooth \(A_{\mathrm{gen}}\) (same knot structure; torque
   bounds as CAP-X2-P0 \(u_{\mathrm{abs,max}}=1.5\) unless a PLAN-X0
   implementation header declares an identical bound before any run).
2. Oracle rollout \(s_0\xrightarrow{A_{\mathrm{gen}}}s_H\).
3. Set \(q^\star=q_H\).

\[
\boxed{\text{each target has at least one feasible open-loop solution}}
\]

Planner / later proposal sees only \(c\), never \(A_{\mathrm{gen}}\).

Cost \(J\) (same structure as CAP-X planning cost; freeze \(\lambda\)):

\[
J(A)=\sum_t
\bigl(\|q_t-q^\star\|^2+\lambda_v\|\dot q_t\|^2+\lambda_u\|u_t\|^2\bigr),
\quad
\lambda_v=0.05,\;\lambda_u=0.01.
\]

No joint-limit violation bonus is required for X0 instrument gates beyond
recording violations; G0 uses feasibility of hidden \(A_{\mathrm{gen}}\).

## Offline teacher (frozen intent)

Do **not** treat \(A_{\mathrm{gen}}\) as the expert.

Warm-start a local optimizer from hidden \(A_{\mathrm{gen}}\) to obtain
\(A^\star\) with

\[
\boxed{J(A^\star)\le J(A_{\mathrm{gen}})}
\]

Teacher may be expensive (offline only). Exact optimizer
(L-BFGS / CEM-refine / gradient-on-oracle) is an **implementation
header lock before the X0 run**, not retuned after seeing \(h\)
histograms. No proposal net is trained in X0.

## Local sensitivity (frozen)

For \(j=1,\ldots,60\), finite-difference curvature at \(A^\star\):

\[
\boxed{
h_j=
\frac{
J(A^\star+\epsilon e_j)+J(A^\star-\epsilon e_j)-2J(A^\star)
}{\epsilon^2}
}
\]

\(\epsilon\) is **not** frozen in this file (PLAN-X1 / X0 implementation
header before run). Ideal local model
\(J(A)\approx J(A^\star)+\frac12\sum_j h_j(A_j-A_j^\star)^2\).
Unnormalized widths \(\tilde\sigma_j^2=1/(h_j+\varepsilon_h)\) with
\(h_j\uparrow\Rightarrow\sigma_j\downarrow\).

Trace matching is **declared** for later X1 (not evaluated as an X0
efficiency claim):

\[
\mathrm{tr}(\Sigma_{\mathrm{sens}})=\mathrm{tr}(\Sigma_{\mathrm{fixed}})=60\sigma_0^2.
\]

X0 only checks that \(h\) is usable (sign, dynamic range).

## Data scale (frozen)

Scene split **inherits CAP-X0 disjoint scene-id convention**
(train \(70000+i\), val \(80000+i\), test \(90000+i\)); PLAN-X uses
**fresh target seeds** (not CAP-X trajectory seeds).

| split | scenes | targets / scene | conditions |
|---|---:|---:|---:|
| train | 128 | 32 | 4096 |
| val | 32 | 32 | 1024 |
| test | 64 | 32 | 2048 |

Artifacts under `runs/plan_x0/` only. Calibration / teacher / \(h\)
fields are oracle-only in HDF5; any later learner pool must exclude
\(A_{\mathrm{gen}}\), \(A^\star\) extras beyond what X1 prereg allows.

## Gates (instrument only)

| id | requirement |
|---|---|
| **G0** | Hidden \(A_{\mathrm{gen}}\): \(S_{\mathrm{feasible}}=1\) (oracle rollout reaches \(q^\star\) within \(\varepsilon_q=0.15\) per-joint max-norm, no construction bug) |
| **G1** | Teacher validity: \(J(A^\star)\le J(A_{\mathrm{gen}})\) on \(\ge 99\%\) of conditions |
| **G2** | Local-min sanity: \(P(h_j>0)\ge 0.90\) over all stored \((condition,j)\) |
| **G3** | Sensitivity diversity: \(Q_{0.9}(h)/Q_{0.1}(h)\ge 5\) on **positive** \(h_j\); else `sensitivity_degenerate` and **PLAN-X1 does not open** |
| **G-label** | `runs/plan_x0/`; no proposal claim; does not write `runs/r10_c0/`; does not modify CAP-X2 artifacts |

If G3 fails: STOP for H2; do not retune \(\epsilon\) after seeing ratios
to force a PASS. A new cell may change \(\epsilon\) only under a new
prereg.

## Claims ceiling

**Allowed if PASS:** the dataset + teacher + \(h\) instrument is closed
enough to freeze PLAN-X1.

**Forbidden:** sampling-efficiency; B2 vs B1; beating CEM; “sensitivity
works”; diffusion; CAP \(R_P\) reinterpretation.

## Unlock

\[
\text{PLAN-X0 formal PASS}
\;\Rightarrow\;
\text{freeze PLANX1\_PREREG.md (architecture, }\epsilon,\,\sigma_0,\,\lambda_h,
\text{seeds, }B\text{ grid, CEM equal-budget protocol)}
\]

## Ledger

```text
PLAN-X0 = PASS  report = REPORT/REP/PLANX/PLANX0_REPORT.md
PLAN-X1 = FROZEN prereg (REPORT/REG/PLANX/PLANX1_PREREG.md)
CAP-X2  = PASS pattern D (orthogonal; do not reopen)
R10     = LOCKED
```

Implementation starts only on explicit request after this freeze
(“开写” / “开始” / “run PLAN-X0”), and must not stop CAP-X2.
