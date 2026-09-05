# PLAN-X1.5 Preregistration — Learned Action Density (not Hessian covariance)

Date: 2026-08-20  
Status: **FROZEN**; **formal FAIL** (`iso_sufficient`; G-mix CI includes 0)  
Depends on: `REPORT/REP/PLANX/PLANX1_REPORT.md` (`anisotropy_no_value`;
H1 supported, H2 rejected)  
Does not: revive \(\Sigma\propto H_J^{-1}\); trigger PLAN-X2
(`proposal_failure` did not occur); diffusion/flow (deferred if mixture
fails on capacity, new cell); CAP-X3; unlock R10; retune PLAN-X1 gates

## Why this cell (not PLAN-X2)

X1 split the original idea:

- **H1** (learn basin center \(\mu_\phi\)) — supported.
- **H2** (shape width by local cost curvature) — rejected, including
  **oracle** \(h\).

PLAN-X2 was reserved for `proposal_failure` (mean not learned). That
did not happen. X1.5 asks the remaining small-model question:

\[
\boxed{
\text{If we do not use Hessian, and instead learn }q(A\mid c)
\text{ as a density, does that beat mean + isotropic }\sigma_0^2 I\text{?}
}
\]

## One question

\[
\boxed{
q_\phi(A\mid c)
\quad\text{vs}\quad
\mathcal N(\mu_\phi(c),\sigma_0^2 I)
\quad\text{at the same rollout grid }B
}
\]

\(c=(q_0,\dot q_0,q^\star,\theta_e)\). Supervise **only** teacher
\(A^\star\) (NLL). **Never** use \(h\) / \(H_J\) in training or sampling.

## Models (frozen)

Shared backbone: **3 × 128 SiLU** (same family as X1).

| id | name | \(q(A\mid c)\) | train |
|---|---|---|---|
| **Iso** | mean + fixed iso (X1-B1 control) | \(\mathcal N(\mu_\phi,\sigma_0^2 I)\), \(\sigma_0=0.25\) | \(\|\mu-A^\star\|^2\) |
| **Diag** | learned diagonal Gaussian | \(\mathcal N(\mu_\phi,\mathrm{diag}(\sigma_\phi^2))\), \(\sigma_j=\mathrm{softplus}\) | Gaussian NLL |
| **Mix** | mixture \(M=4\) (primary density) | \(\sum_{m=1}^{4}\pi_m\mathcal N(\mu_m,\mathrm{diag}(\sigma_m^2))\) | mixture NLL |

No trace-matching to Hessian. Diag/Mix may choose their own scale
(the point of density learning). Samples clipped to \(\lvert A_j\rvert\le 1.5\).

**No diffusion.** If Mix fails because 4 diagonal Gaussians cannot
represent \(p(A\mid c)\), that is a **new** cell, not a silent upgrade.

## Protocol

| item | frozen |
|---|---|
| data | `runs/plan_x0/formal/` (same splits) |
| seeds | \(\{301,302,303,304,305\}\) |
| AdamW / lr / wd / batch / 40 ep / patience 8 | same as X1 |
| \(B\) | \(\{16,32,64,128,256\}\) |
| success | \(\|q_T-q^\star\|_\infty<0.15\) |
| primary | \(AUC_S=\mathrm{AUC}_{\log B}S(B)\) |
| CEM | \(B_{\mathrm{tot}}\in\{128,256,512,1024\}\), 4 iters, elite 0.10 |

CEM variants (equal rollouts): **vanilla** (broad start), **Iso-seed**,
**Mix-seed**. First iteration draws from that proposal; later iters
refit as in X1.

5-seed mean of per-condition curves; 95% paired bootstrap CI across
2048 test conditions.

## Gates

| id | requirement |
|---|---|
| **G-H1** | Iso still beats zero-mean action: \(J(\mu_{\mathrm{iso}})<J(0)\), CI \(>0\) (H1 intact) |
| **G-iso** | \(AUC_S^{\mathrm{Iso}}-AUC_S^{\mathrm{broad}}>0\), CI \(>0\) (control vs \(\mathcal N(0,0.75^2 I)\)) |
| **G-mix** | \(AUC_S^{\mathrm{Mix}}-AUC_S^{\mathrm{Iso}}>0\), CI \(>0\) (**core**) |
| **G-cem** | Mix-seed CEM \(AUC_S\) \(>\) Iso-seed CEM, CI \(>0\) (density helps refinement, not only open-loop samples) |
| **G-label** | `runs/plan_x15/`; no \(h\) in the sampler; no diffusion; no R10 |

Diag vs Iso is **logged**, not a pass gate: if Diag wins and Mix does
not, report `diag_sufficient`. G-cem failure with G-mix pass is
allowed (`openloop_density_only`).

`plan_x15_passed` iff G-H1 \(\land\) G-iso \(\land\) G-mix \(\land\) G-label
(G-cem is reported, not required for PASS — open-loop density is the
cell question; CEM is secondary).

## Patterns (do not retune)

| pattern | meaning |
|---|---|
| `density_beats_iso` | G-mix PASS — learned \(q\) beats mean+iso |
| `diag_sufficient` | Diag CI \(>0\) vs Iso, Mix not better than Diag |
| `iso_sufficient` | Mix \(\approx\) Iso (CI includes 0 or Mix worse); mean+iso is enough |
| `density_hurts` | Mix CI \(<0\) vs Iso |
| `h1_broken` | G-H1 fail (unexpected; stop) |

## Claims ceiling

**Allowed:** on this oracle reachable-target host, learned action
density (mixture / diag) does or does not beat X1’s winning isotropic
proposal.

**Forbidden:** resurrecting H2; calling this PLAN-X2; diffusion
success; CAP \(R_P\) mixing; R10.

## Ledger

```text
PLAN-X1   = FAIL anisotropy_no_value (H1 yes, H2 no)
PLAN-X1.5 = FAIL  iso_sufficient (REPORT/REP/PLANX/PLANX15_REPORT.md)
PLAN-X2   = LOCKED not triggered
CAP-X3    = LOCKED
R10       = LOCKED
```
