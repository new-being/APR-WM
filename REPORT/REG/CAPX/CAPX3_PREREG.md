# CAP-X3 Preregistration — Low-Dimensional Scene Adaptation

Date: 2026-08-21  
Status: **FROZEN**; **P0 PASS**; **formal PASS**
(`structured_adaptation_advantage`; \(K_{90}^{\mathrm{phy}}=1\),
\(K_{90}^{\mathrm{lat}}=2\), \(R_K=0.5\))  
Depends on: `REPORT/REP/CAPX/CAPX3_P0_REPORT.md` (`cap_x3_p0_passed=true`)  
Does not: CAP-X2 residual \(\rho>0\) (that is **CAP-X3B**, locked);
CAP-X4 visual; diffusion; PLAN-X2; unlock R10; reuse CAP-X0/X1/X2
formal test scenes as adaptation targets

## Why this cell (not PLAN diffusion, not CAP-X2 rescue)

PLAN-X closed the action-prior branch on this host:

\[
\boxed{
\text{learn where to search is supported;}
\text{more complex }q(A\mid c)\text{ has no extra significant value.}
}
\]

Working proposal remains \(\mathcal N(\mu_\phi(c),\sigma_0^2 I)\).
Diffusion stays locked until a **declared multimodal** task family exists.

CAP-X1: matched physics can drive learned dynamics capacity to 0
(\(R_P(0)=1\) upper bound). CAP-X2: off-family ruler broke
(`reference_failure`); do not widen PureNN to fake \(R_P(\rho)\).

The original question still unanswered:

\[
\boxed{
\textbf{Does causal parameterization reduce the amount of new-scene data
required for adaptation, beyond merely restricting adaptation to a
low-dimensional context?}
}
\]

中文：物理因果参数化的适应优势，是否超过“只是把可调自由度压低”本身？

## One question

Given a **held-out** robot scene and a small calibration set \(D_{\mathrm{cal}}^{(K)}\),
which small parameter block restores prediction fastest?

\[
\boxed{
\theta_E
\quad\text{(structured physics)}
\quad\text{vs}\quad
z_E\in\mathbb R^{d_\theta}
\quad\text{(same-dim unstructured latent)}
}
\]

Not: “can physics fit this system?” (X0/X1).  
Not: “how does \(R_P\) decay with \(\rho\)?” (X2, closed).

## Protocol freeze (formal; P0 may only shrink active \(\theta\))

| item | frozen |
|---|---|
| host | `capx_arm3.v1`, \(\rho=0\), \(\mu=0\) |
| \(T_{\mathrm{cal}}\) | **\(1.0\,\mathrm{s}\)** (do not also sweep length) |
| \(K\) | \(\{1,2,4,8,16\}\) |
| query | 8 held-out traj × \(1.0\,\mathrm{s}\) / scene (never in \(D_{\mathrm{cal}}\)) |
| meta-train / val / test scenes | 128 / 32 / 64 **fresh** seeds \(\{110000,120000,130000\}+i\) |
| \(d_z\) | \(d_z=d_{\theta,\mathrm{active}}\) (P0 freeze; default 10, \(\mu\) slots dropped) |
| same \(D_{\mathrm{cal}}^{(K)}\) | physics, latent, and fine-tune **all see the same transitions** |
| \(H\) rollout | \(\{10,50,100\}\) learner steps (100 Hz) |

Do **not** reuse scene seeds \(70\mathrm{k}/80\mathrm{k}/90\mathrm{k}\) (CAP-X0).

### A. Physics-parameter adaptation

\(\Psi_{\mathrm{phy}}\) frozen. Fit only active \(\theta_E\) by one-step
\(\|\hat{\ddot q}_{\mathrm{phy}}-{\ddot q}\|^2\) on \(D_{\mathrm{cal}}^{(K)}\).
No neural residual. No shared-code update.

### B. Same-dimensional latent

\(F_\psi(q,\dot q,u,z_E)\), \(z_E\in\mathbb R^{d_\theta}\). Meta-train:
shared \(\psi\) and **per-scene** \(z_i\) jointly; **no \(\theta\) labels**
into \(z\). New scene: freeze \(\psi\), fit \(z_E\) on the same
\(D_{\mathrm{cal}}^{(K)}\).

### C. Full fine-tune ceiling (not the main contrast)

Unfreeze \(\psi\) and \(z_E\) (or PureNN without scene vector). Reports
adaptation **ceiling**, not \(R_K\).

### Secondary diagnostic (not a pass gate)

Random orthogonal \(T\) on \(\theta\) (`rotated_theta`): logged only.
Core contrast is A vs B.

## Metrics

- **M1** \(E_1(K)\): NRMSE\((\hat{\ddot q},\ddot q)\) on query.
- **M2** \(E_{\mathrm{roll}}(K)\) at \(H\in\{10,50,100\}\).
- **M3** \(E_\theta(K)\): \(\|\hat\theta-\theta\|/\|\theta\|\) physics only
  (mechanism, not cross-method primary).

Primary summary:

\[
K_{90}
=
\min\bigl\{K\in\{1,2,4,8,16\}:
E_1(K)\le E_1(16)+0.1\bigl(E_1(0)-E_1(16)\bigr)
\bigr\}
\]

\(E_1(0)\): no-calibration baseline (physics at nominal \(\theta\);
latent at \(z=0\)). If no \(K\) qualifies, \(K_{90}:=16^+\) (fails
sample-efficiency claim).

\[
R_K=1-\frac{K_{90}^{\mathrm{phy}}}{K_{90}^{\mathrm{latent}}}
\]

Paired bootstrap across test scenes on \(K_{90}\) ranks / \(E_1(K)\) curves.

## Patterns (do not retune after seeing \(R_K\))

| pattern | meaning |
|---|---|
| `structured_adaptation_advantage` | \(K_{90}^{\mathrm{phy}}<K_{90}^{\mathrm{latent}}\), paired CI on \(\Delta E_1\) or rank clear |
| `dimension_only` | \(K_{90}\) tied, both beat full fine-tune sample-wise |
| `latent_advantage` | latent \(K_{90}\) smaller |
| `identification_failure` | physics \(E_\theta\) stays large **and** \(E_1\) does not recover |

**Only Pattern D** reopens R8/R9-style active identification.  
CAP-X3B (\(\theta_E+z_{\mathrm{res}}\) at \(\rho>0\)) only if X3 is not
`identification_failure` and a new prereg is frozen.

`cap_x3_passed` iff P0 passed and pattern is A, B, or C with G-label
(disjoint seeds, \(\rho=0\), equal \(D_{\mathrm{cal}}\), \(d_z=d_\theta\)).

## Claims ceiling

**Allowed:** on this matched 3-DoF host, whether causal \(\theta_E\)
beats same-dim latent on \(K_{90}/R_K\).

**Forbidden:** \(R_P(\rho)\) rescue; visual; diffusion; R10; claiming
fine-tune ceiling as the main physics win.

## Ledger

```text
PLAN-X   = closed on this host (μ+iso sufficient)
CAP-X2   = COMPLETE reference_failure (do not rescue)
CAP-X3-P0 = PASS
CAP-X3    = PASS  structured_adaptation_advantage
CAP-X3B   = LOCKED
CAP-X4    = LOCKED
R10       = LOCKED
```
