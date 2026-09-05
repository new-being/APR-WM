# APR-WM Paper Architecture — Three Pillars After CAP-X3

Date: 2026-08-23  
Status: **STORYLINE FREEZE** for CAP/PLAN/validity pillars; **TASK-X
family opened 2026-08-27**; **TASK-X0 RAN / family FAIL**; **TASK-XL
FROZEN after `instrument_ready`** (learned \(z_g,z^p\); not run; see addendum)  
Depends on: CAP-X0–X3; PLAN-X0–X1.5; R7–R9 + SIM-X + REAL-LOG-A0  
Does not: open CAP-X3B, CAP-X4, PLAN-X2 on `capx_arm3`, R10; retune
frozen CAP/PLAN gates. **Addendum 2026-08-27:** TASK-X is a **new**
family (`REPORT/REG/TASKX/TASKX_PREREG.md`); TASK-X0 trains **no**
diffusion and **did not PASS** (instrument / representation FAIL, not
“progress has no planning value”). TASK-X1 stays **LOCKED**. **TASK-XL**
**RTWX-X0S RAN / `timing_mismatch`**; **X0F RAN / `force_channel_unresolved`**.
Do **not** reopen PLAN-X2.

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

## Addendum 2026-08-27 — TASK-X (not a fourth paper pillar yet)

Opened as a **separate prereg family**, orthogonal to RoboTwin-X0R
(dynamics repair; **NEXT / highest IG**) and to PLAN-X2 (still locked
on the unimodal arm).

\[
\boxed{
p_t,g\rightarrow\text{action selection},\quad
p_t,g\nrightarrow F_{\mathrm{physics}}
}
\]

TASK-X0 **RAN; family FAIL**. Honest reading:

\[
\boxed{
\textbf{instrument / representation test did not PASS;}
\text{ this is not ``task progress has no planning value.''}
}
\]

Keep two classes: cabinet `coverage_hole` = instrument insufficient
(not “\(p_t\) has no decision value”); cup/stamp `p_no_nll` = this
explicit \(p_t\) did not further compress action uncertainty. PLAN-X
already showed state+goal can learn a search center; TASK-X0 did **not**
show explicit progress further improves the action distribution.
Remaining **closed**: diffusion, TASK-X1. Do **not** rewrite “progress
unproven” as “need a stronger generator.”

TASK-X1 (oracle \(p\) + matched attractors) stays locked. Visual
\(\hat p\) is TASK-X2, locked. **TASK-XL** (2026-08-27 freeze):
hand phase tables are **not** the long-run representation. Split
\(z_g=E_g(g)\) (slow task latent) and
\(z^p_t=E_p(h_t,z_g)\) with \(h_t=(s_{0:t},a_{0:t-1})\) (fast;
not \(f(s_t)\) alone — task–state aliasing). Primary test:
\(H(A\mid s,z_g,z^p)<H(A\mid s,z_g)\). Encoder causal:
\(E_p(s_{\le t},a_{<t},g)\) only. State box:
\(X_t=(s^{\mathrm{phy}}_t,\theta_E,b^{\mathrm{obs}}_t,z_g,z^p_t,z^{\mathrm{res}}_t)\);
\(F_{\mathrm{physics}}\) physics-only;
\(q(A\mid\cdot)\) eats \((s^{\mathrm{phy}},z_g,z^p)\).
Retry test = **learned progress vs no-progress**, not rewrite phases.
Cabinet remains an instrument hole; cup/stamp kNN G2 does not prove
learned \(z^p\) worthless. XL is **LOCKED until X0R / object-state
demos**. Diffusion still only after instrument PASS.

Priority:
X0R \(>\) TASK-X retry \(>\) visual \(>\) diffusion. If TASK-X is
retried later: first fix coverage and object-state logging, not swap
models / not open TASK-X1.

This does **not** convert PLAN-X `iso_sufficient` into “use diffusion
on `capx_arm3`.”

## What not to open now

CAP-X3B, frozen CAP-X4 visual, PLAN-X2 on the arm host, R10, TASK-X1,
TASK-X2, TASK-XL0 train, RoboTwin-X1 RGB. Next is **RTWX-X0S**, not
capacity or XL.

## Pointers

- CAP-X3: `REPORT/REP/CAPX/CAPX3_REPORT.md`
- PLAN-X1.5: `REPORT/REP/PLANX/PLANX15_REPORT.md`
- TASK-X family: `REPORT/REG/TASKX/TASKX_PREREG.md`
- RTWX-X0S: `REPORT/REG/RTWX/robot_dynamics/RTWX0S_PREREG.md`
- RTWX index: `REPORT/REG/RTWX/README.md`

## Addendum 2026-08-30 — SYM-X object memo (not a fourth paper pillar yet)

SYM-X0–X2 **RAN** on oracle rigid state. Object coordinates are no
longer a fixed full \(R\):

\[
m_i=(\bar s_i,\gamma_i,b_i(G)),\qquad
D_H\to b(G)\to\ell(\gamma)\to\text{allocation}.
\]

\(F_{\mathrm{physics}}\) follows \(\alpha_\gamma^{\mathrm{world}}\);
task may follow \(\alpha_\gamma^{\mathrm{task}}\). Prediction residual
is **not** a sufficient test of causal abstraction (X2:
\(E_{\mathrm{stale}}/E^{B0}=1.002\) while \(D_H(R_y90)=6.44\)).
Contract: `REPORT/REG/SYMX/SYMX_MECHANISM.md`. **SYM-X3 / RGB stays
LOCKED.** Do not claim FLOPs reduction. Do not reopen O0G6R.
