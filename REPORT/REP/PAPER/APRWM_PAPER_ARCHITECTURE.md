# APR-WM Paper Architecture — Three Pillars After CAP-X3

Date: 2026-08-23  
Status: **STORYLINE FREEZE** (no new experiment cell)  
Depends on: CAP-X0–X3; PLAN-X0–X1.5; R7–R9 + SIM-X + REAL-LOG-A0  
Does not: open CAP-X3B, CAP-X4, PLAN-X2, diffusion, R10; retune frozen gates

## One sentence

\[
\boxed{
\textbf{APR-WM = structured priors + selective learning + validity control}
}
\]

Not a larger world model. Structure compresses **what must be learned**;
learning is **local** (scene \(\theta_E\), action mean \(\mu_\phi\));
validity decides **when that structure is still trustworthy**.

## Three pillars (paper figure)

| pillar | question | supported claim | not supported |
|---|---|---|---|
| **Dynamics** (CAP-X) | Does causal structure reduce model freedom *and* adaptation cost? | X1: matched physics can replace PureNN capacity (\(R_P(0)=1\) upper bound). X3: same \(d=10\), causal \(\theta_E\) beats unstructured \(z_E\) (\(K_{90}=1\) vs \(2\), \(R_K=0.5\)). | X2: off-family \(R_P(\rho)\) **undefined** (`reference_failure`); do not rescue with a wider PureNN. |
| **Action** (PLAN-X) | Does experience compress search? | H1: learn **where** to search (\(\mu_\phi\)). Working proposal \(\mathcal N(\mu_\phi,\sigma_0^2 I)\). | H2 Hessian covariance; Mix-4 / diag density; diffusion **not** a natural upgrade on this unimodal host. |
| **Validity** (R7–R9 / SIM-X) | When is the structure still licensed? | Positive certification is possible; validity has a lifetime; **policy-induced epistemic attenuation** is cross-engine. | Persistent epistemic closure (SIM-X3); real-force residual (R10 **unmade**). |

## Contribution points (draft)

1. **Capacity substitution under matched family** (CAP-X1): explicit
   rigid-body structure can drive learned dynamics capacity to zero at
   \(\rho=0\). Wording: *upper bound*, not “physics always smaller.”
2. **Adaptation efficiency beyond dimension** (CAP-X3): equal \(d\),
   equal \(D_{\mathrm{cal}}\), causal coordinates need less new-scene
   data. This is the original “invariant causal representation” hit,
   not “physics has fewer knobs.”
3. **Mismatch honesty** (CAP-X2): if the neural ruler is incompetent,
   \(R_P(\rho)\) is not a number. Do not convert Pattern D into
   “physics lost at \(\rho=0.05\).”
4. **Search location, not search shape** (PLAN-X): amortize planning
   experience into \(\mu_\phi(c)\); isotropic local sampling is enough
   here. Do not inflate into a generative action model.
5. **Validity is policy-conditioned** (R7–R9, SIM-X): certificates can
   authorize deviation from a robust default; evidence rate depends on
   the action; real-physics residual remains **explicitly unmade**.

## Hypothesis split (CAP-X3 only)

| id | claim | status |
|---|---|---|
| A | Advantage is only \(d=10\) | **excluded** (\(d_z=d_\theta\), curves differ) |
| B | Causal axes have better identification geometry | compatible (not isolated) |
| C | Shared mechanism is invariant across scenes | compatible (not isolated) |

Do not write “B proven, C proven.” Write **A excluded; coordinate
semantics matter**.

## Suggested paper spine

```text
1. Problem: scene change + limited calibration + when to trust WM
2. Dynamics prior  → CAP-X1 capacity, CAP-X3 few-shot (X2 as boundary)
3. Action prior    → PLAN-X μ+iso (negative: Hessian / density)
4. Validity        → R7 certificate, R9 lifetime, SIM-X attenuation
5. Unmade          → R10-C0 / REAL-LOG-A0 data-source lock
```

## What not to open now

CAP-X3B (\(\rho>0\) joint \(\theta+z_{\mathrm{res}}\)), frozen CAP-X4
(visual front-end), PLAN-X2 / diffusion, R10, active probe (Pattern D
did not fire).

If experiments resume later, the **new** question is residual *vs*
causal split — **new prereg**, not a silent rename of CAP-X4 (family
CAP-X4 remains visual and locked).

## Pointers

- CAP-X3: `REPORT/REP/CAPX/CAPX3_REPORT.md`
- PLAN-X1.5: `REPORT/REP/PLANX/PLANX15_REPORT.md`
- Validity synthesis: `REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md`
