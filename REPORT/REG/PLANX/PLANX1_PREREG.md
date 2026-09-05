# PLAN-X1 Preregistration — Unimodal Sensitivity-Aware Gaussian Proposal

Date: 2026-08-18  
Status: **FROZEN**; **formal FAIL** (`plan_x1_passed=false`);
pattern **`anisotropy_no_value`** — H2 STOP; do not retune gates
depends on `REPORT/REP/PLANX/PLANX0_REPORT.md` (`plan_x0_passed=true`)  
Does not: diffusion; mixture \(M>1\); world-model uncertainty (X3);
retune X0 \(\epsilon_h\); reopen CAP-X2-P0; unlock R10; claim H2 from
X0 \(h\) diversity alone

## One question

\[
\boxed{
\text{At equal total exploration }\mathrm{tr}(\Sigma)=60\sigma_0^2,
\text{ does sensitivity-shaped }\Sigma\text{ beat isotropic }\Sigma
\text{ around the same learned mean?}
}
\]

H1 = learned attractor location \(\mu_\phi(c)\).  
H2 = trace-matched sensitivity covariance.  
Core contrast: **B2 vs B1**.

## Data / condition (from X0)

Reuse `runs/plan_x0/formal/` splits. Learner sees only

\[
c=(q_0,\dot q_0,q^\star,\theta_e)\in\mathbb{R}^{22}.
\]

Supervises on oracle `A_star`, `h` (train/val). Test metrics may use
oracle \(h\) **only** for B3 diagnostic ceiling.

## Architecture (frozen)

Shared MLP: **3 × 128 SiLU**, then two heads:

- \(\mu_\phi(c)\in\mathbb{R}^{60}\)
- \(\hat h_\phi(c)\in\mathbb{R}_+^{60}\) via softplus

\[
\hat\sigma_j^2
=
c\cdot\frac{1}{\hat h_j+\varepsilon_h},
\qquad
\sum_j\hat\sigma_j^2=60\sigma_0^2.
\]

\(\varepsilon_h=10^{-6}\). Carry X0 \(\epsilon_h=0.05\) for any on-policy
FD checks; do not retune.

## Training (frozen)

| item | value |
|---|---|
| \(\mathcal L\) | \(\|\mu-A^\star\|^2+\lambda_h\,\mathrm{Huber}(\log\hat h,\log h^+)\) |
| \(\lambda_h\) | 1.0 |
| \(h^+\) | \(\max(h,10^{-8})\) |
| optimizer | AdamW, lr \(10^{-3}\), wd \(10^{-4}\) |
| batch | 256 |
| epochs | 40, cosine, patience 8 on val \(\mathcal L\) |
| seeds | \(\{201,202,203,204,205\}\) |
| \(\sigma_0\) | 0.25 (absolute, on knot coordinates; \(\lvert u\rvert\le 1.5\)) |
| \(\Sigma_{\mathrm{broad}}\) | \(\sigma_{\mathrm{broad}}^2 I\) with \(\sigma_{\mathrm{broad}}=0.75\) |

No per-seed width search. No diffusion.

## Baselines (frozen)

| id | sampler |
|---|---|
| **B0** | \(\mathcal N(0,\Sigma_{\mathrm{broad}})\) |
| **B1** | \(\mathcal N(\mu_\phi,\sigma_0^2 I)\) |
| **B2** | \(\mathcal N(\mu_\phi,\Sigma_{\mathrm{sens},\phi})\) primary |
| **B3** | same \(\mu_\phi\), oracle FD \(h\) → trace-matched \(\Sigma\) (**ceiling only**) |

## Equal-cost geometry (before success)

At \(A^\star\), 1\(\sigma\) coordinate probes. \(CV_{\Delta J}=\mathrm{std}_j(\Delta J_j)/\mathrm{mean}_j(\Delta J_j)\).

\[
\boxed{CV_{\mathrm{sens}}\le 0.7\,CV_{\mathrm{fixed}}}
\]

(B2 predicted \(\sigma\); B1/fixed uses \(\sigma_0\). Evaluate on test,
mean over conditions.)

## Sampling efficiency

\(B\in\{16,32,64,128,256\}\) oracle rollouts; \(A^*=\arg\min_k J(A^{(k)})\).

\[
AUC_S=\mathrm{AUC}_{\log B}\,S(B),
\quad
S=\mathbb{P}(\|q_T-q^\star\|_\infty<\varepsilon_q=0.15).
\]

Also log \(J_{\mathrm{best}}(B)\). Primary: **B2 vs B1** paired 95%
bootstrap CI of \(AUC_S^{B2}-AUC_S^{B1}\) over test conditions
(5-seed mean of \(AUC_S\) first, then CI across conditions).

## Equal-budget CEM (second experiment)

\(B_{\mathrm{total}}\in\{128,256,512,1024\}\). Vanilla CEM: 4 iters,
\(N_{\mathrm{cand}}=B_{\mathrm{total}}/4\). Proposal-CEM: first iter
from \(q_\phi=\) B2, then same CEM refit. Elite frac 0.10. Same \(J\),
horizon, knots as X0.

\[
AUC_S^{\mathrm{proposal\text{-}CEM}}>AUC_S^{\mathrm{vanilla\text{-}CEM}}
\]

paired 95% CI \(>0\).

Optional summary (not a gate): \(B_{80}=\min B:S(B)\ge0.80\),
\(R_B=1-B_{80}^{\mathrm{sens}}/B_{80}^{\mathrm{CEM}}\) if both exist.

## Gates

| id | requirement |
|---|---|
| **G0** | mean gen: \(J(\mu_\phi)<J(0)\) paired CI \(>0\) on test |
| **G1** | Spearman \((\log\hat h,\log h)>0\) and 95% CI \(>0\) (positive \(h\)) |
| **G2** | \(CV_{\mathrm{sens}}\le 0.7\,CV_{\mathrm{fixed}}\) |
| **G3** | \(AUC_S^{B2}>AUC_S^{B1}\), paired 95% CI \(>0\) (**core**) |
| **G4** | proposal-CEM \(AUC_S\) \(>\) vanilla CEM, paired CI \(>0\) |
| **G-label** | `runs/plan_x1/`; no diffusion; no R10 |

All PASS → `sensitivity_aware_proposal_success`.

## Failure patterns (do not retune to escape)

| pattern | meaning |
|---|---|
| `mean_only_success` | B1 ≫ B0 but B2 \(\approx\) B1 |
| `sensitivity_prediction_failure` | B3 helps, B2 does not |
| `anisotropy_no_value` | B2 and B3 both fail vs fixed |
| `proposal_failure` | learned mean not better than B0 |

`anisotropy_no_value` → STOP H2. `proposal_failure` → PLAN-X2 multimodal,
not bigger MLP.

## Claims ceiling

**Allowed if PASS:** unimodal sensitivity-aware proposal reduces
rollouts vs isotropic exploration around the same mean (and vs CEM if
G4).

**Forbidden:** diffusion success; CAP \(R_P\) mixing; real-robot
planning; treating X0 diversity as H2 confirmation.

## Ledger

```text
CAP-X0       = PASS
CAP-X1       = PASS
CAP-X2       = COMPLETE  pattern=reference_failure
CAP-X3       = LOCKED

PLAN-X0      = PASS
PLAN-X1      = FAIL  pattern=anisotropy_no_value
PLAN-X1.5    = FAIL  iso_sufficient; not a retune of X1 gates
H1           = SUPPORTED
H2           = REJECTED on this host/task
PLAN-X2      = LOCKED (not triggered)
R10          = LOCKED
```
