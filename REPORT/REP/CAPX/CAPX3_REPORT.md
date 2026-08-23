# CAP-X3 Report — Low-Dimensional Scene Adaptation

Date: 2026-08-22  
Status: **`cap_x3_passed=true`**; pattern **`structured_adaptation_advantage`**  
Prereg: `REPORT/REG/CAPX/CAPX3_PREREG.md` (gates **not** retuned)  
P0: `REPORT/REP/CAPX/CAPX3_P0_REPORT.md`  
Data: 128/32/64 fresh scenes (seeds \(110\mathrm{k}/120\mathrm{k}/130\mathrm{k}\))  
Artifacts: `runs/cap_x3/formal/`  
Does not: \(\rho>0\); CAP-X2 rescue; active probe; PLAN-X; unlock R10;
claim fine-tune ceiling as the physics win

## Question

On a **held-out** matched scene, with the same \(D_{\mathrm{cal}}^{(K)}\),
does causal \(\theta_E\in\mathbb R^{10}\) reach 90% of its own
full-calibration gain with fewer trajectories than a same-dim latent
\(z_E\)?

\[
K_{90}
=
\min\{K:E_1(K)\le E_1(16)+0.1(E_1(0)-E_1(16))\}
\]

## Pattern (pre-declared)

\[
\boxed{\texttt{structured\_adaptation\_advantage}}
\]

\[
K_{90}^{\mathrm{phy}}=1,\qquad
K_{90}^{\mathrm{lat}}=2,\qquad
R_K=0.5
\]

Per-scene \(K_{90}^{\mathrm{phy}}-K_{90}^{\mathrm{lat}}\): mean
\(-0.64\), 95% CI \((-1.13,-0.03)\) **entirely below 0**.

Causal coordinates are not “just 10 scalars.” They adapt with less
new-scene data than an equally low-dimensional unstructured context.

\[
\boxed{
\textbf{causal parameterization provides adaptation efficiency
beyond dimensionality reduction}
}
\]

Not: “physics has fewer parameters.” Both adapters have \(d=10\).
The gap is **coordinate semantics** (causal axes vs arbitrary
\(z\to F\) axes), not dimension count.

## M1 — \(E_1(K)\) (query NRMSE \(\ddot q\); 64 scenes)

| \(K\) | Physics \(\theta_E\) | Latent \(z_E\) | Full ft ceiling |
|---:|---:|---:|---:|
| 0 (no cal) | 0.339 | 0.349 | — |
| 1 | **0.0076** | 0.102 | 0.206 |
| 2 | 0.0095 | 0.080 | 0.144 |
| 4 | 0.0056 | 0.069 | 0.075 |
| 8 | 0.0073 | 0.068 | 0.061 |
| 16 | 0.0055 | 0.067 | 0.051 |

Physics is essentially done at **\(K=1\)** (one 1 s trajectory). Latent
keeps a high floor \(\approx 0.067\) even at \(K=16\). Full fine-tune is
the slowest at small \(K\) (more free parameters) and still does not
match physics \(E_1\) at \(K=16\).

This is **not** Pattern B (`dimension_only`): same \(d=10\) does not
equalize the curves. Hypothesis A (low dimension alone) is **excluded**
on this host. B (optimization geometry / causal alignment) and C
(invariant mechanism) remain jointly compatible; they are not split
here.

Full fine-tune is slower at small \(K\): more trainable weights are
**not** faster adaptation.

## M3 — \(\theta\) recovery vs prediction

| \(K\) | \(E_\theta\) | Physics \(E_1\) |
|---:|---:|---:|
| 1 | 0.090 | 0.0076 |
| 16 | 0.084 | 0.0055 |

Spearman\((-E_\theta,-E_1^{\mathrm{phy}})\) on the mean \(K\)-curve:
**0.90**. Parameter error and prediction error drop **together** at
\(K=1\). That is the full chain (identifiable \(\theta\) \(\to\) matched
family), not “\(\theta\) recovered but \(F_{\mathrm{phy}}\) wrong.”

## M2 — rollout

Query length is \(1.0\,\mathrm{s}\) at 100 Hz (100 steps), so
\(H=100\) never fits (`NaN`; instrument, not a retune). \(H\in\{10,50\}\):

| \(K=16\) | \(H=10\) | \(H=50\) |
|---|---:|---:|
| Physics | 0.136 | 0.326 |
| Latent | 0.178 | 0.493 |
| Full | 0.148 | 0.387 |

Same ranking as \(E_1\). Not a \(R_K\) input.

## What this does **not** mean

- Not CAP-X2 rescued: this is \(\rho=0\) matched family.
- Not “latent cannot learn”: \(z\) beats \(z=0\) (P0 and the \(K\) curve).
  It is **less sample-efficient** and **worse asymptotically** here.
- Not a license for active probe (Pattern D did not occur).
- Not CAP-X3B / visual / R10.

## Claims ceiling

**Allowed:** on this 3-DoF matched host, causal \(\theta_E\) yields
few-shot adaptation efficiency **beyond** same-dim latent reduction
(\(R_K=50\%\) on frozen \(K_{90}\)).

**Forbidden:** “physics wins because it has fewer parameters”;
\(R_P(\rho)\) rescue; diffusion; R10; treating \(H=100\) NaN as a
physics failure; opening CAP-X3B from this PASS.

## Ledger

```text
CAP-X3-P0 = PASS
CAP-X3    = PASS  structured_adaptation_advantage
            K90_phy=1, K90_lat=2, R_K=0.5
CAP-X3B   = LOCKED
CAP-X4    = LOCKED
R10       = LOCKED
PLAN-X    = frozen after iso_sufficient
```
