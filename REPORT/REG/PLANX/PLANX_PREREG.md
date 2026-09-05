# PLAN-X Preregistration — Sensitivity-Aware Learned Action Proposal

Date: 2026-08-18  
Status: **FROZEN** (family); **PLAN-X0 PASS**; **PLAN-X1 FAIL**
(`anisotropy_no_value`: H1 supported, H2 rejected); **PLAN-X1.5 FAIL**
(`iso_sufficient`: Mix-4 does not beat mean+iso); **PLAN-X2 LOCKED
(not triggered)**; action prior closed as \(\mu+\sigma_0^2 I\); **CAP-X3
opened at P0**; does not unlock R10  
Depends on: host `capx_arm3.v1`; CAP-X2-P0 diagnosis (planning
instrument insufficient — orthogonal motivation, not a CAP-X retune)  
Does not: mix into CAP-X \(R_P(\rho)\); retune CAP-X2-P0 CEM budget;
start from diffusion; claim planning-equivalent dynamics replacement;
RoboCasa as first PLAN host

## Orthogonality

\[
\boxed{
\text{CAP-X answers dynamics capacity. PLAN-X answers search capacity.}
}
\]

CAP-X2 remains **predictive-dynamics only** (\(M=M_1\land M_2\)). PLAN-X
does **not** reopen P0, enlarge CEM grids, or rewrite X1/X2 matched
gates. The P0 fact \(S_{\mathrm{oracle}}=0.708<0.80\) at max frozen
budget (41.83 s/task) is motivation that **proposal/search** has slack;
it is not a license to unfreeze P0.

Paper symmetry (allowed as framing; not an X0 claim):

\[
\boxed{
\begin{array}{ll}
\text{CAP-X:} &
\text{structured physics compresses dynamics-model freedom}\\[1mm]
\text{PLAN-X:} &
\text{action priors compress planning-search freedom}
\end{array}
}
\]

## One question

\[
\boxed{
\textbf{能否把历史规划经验摊销成条件动作采样分布，
并利用动作}\to\text{结果敏感性自适应分配采样方差，
从而显著减少在线 world-model rollout 数？}
}
\]

Two hypotheses, **separable** (must not start with diffusion):

\[
\boxed{\textbf{H1: experience learns where to search}}
\]

\[
\boxed{\textbf{H2: sensitivity learns how wide to search per direction}}
\]

If a unimodal Gaussian already moves the success-vs-rollouts curve
left, multimodal / diffusion is deferred.

## Route (frozen)

```text
PLAN-X0  proposal benchmark + teacher/sensitivity instrument
   ↓
PLAN-X1  single-mode sensitivity-aware Gaussian proposal
   ↓
PLAN-X1.5 learned q(A|c) vs mean+iso (not Hessian; not PLAN-X2)
   ↓
PLAN-X2  multimodal attractors: mixture / diffusion (only if needed)
   ↓
PLAN-X3  model uncertainty + sensitivity (APR-WM, not oracle)
   ↓
PLAN-X4  full APR-WM + learned proposal
```

| Cell | Question | When |
|---|---|---|
| **PLAN-X0** | Are reachable targets, teacher \(A^\star\), and FD curvature \(h\) a fair instrument? | prereg frozen now |
| **PLAN-X1** | Does learned \(\mu\) (H1) plus trace-matched \(\Sigma_{\mathrm{sens}}\) (H2) beat isotropic \(\Sigma\) at equal rollout budget? | freeze **after** X0 PASS |
| **PLAN-X1.5** | If not Hessian, does learned \(q(A\mid c)\) (Mix-4 / Diag) beat mean+\(\sigma_0^2 I\)? | FAIL `iso_sufficient` |
| **PLAN-X2** | When \(p(A\mid c)\) is multimodal, does a small mixture suffice before diffusion? | locked |
| **PLAN-X3** | Sensitivity \(\neq\) epistemic uncertainty; how to gate \(\Sigma\) by WM validity? | locked |
| **PLAN-X4** | Full stack | locked |

**Do not preregister PLAN-X1 network width, \(\epsilon\), \(\sigma_0\),
train budget, or formal seeds until PLAN-X0 PASS.** Family document
may sketch X1 comparisons; those knobs stay unlocked.

## Host / action object (X0–X1)

\[
\boxed{\texttt{capx\_arm3.v1}}
\]

3-DoF, no contact, **oracle dynamics** (no world-model error in X0/X1).
Horizon \(1\,\mathrm{s}\), 20 action knots, piecewise-constant torques:

\[
A\in\mathbb{R}^{60}.
\]

Condition (X0/X1; \(\theta_e\) oracle-visible — proposal geometry only):

\[
\boxed{c=(q_0,\dot q_0,q^\star,\theta_e)}
\]

PLAN-X3 replaces \(\theta_e\) by \(p(\theta_e\mid\mathcal H_t)\).

## Intended X1 comparisons (not frozen)

After X0 PASS, X1 is expected to compare (same mean backbone when
applicable; **equal trace** \(\mathrm{tr}(\Sigma)=60\sigma_0^2\)):

| id | sampler |
|---|---|
| **B0** | Broad prior \(\mathcal N(0,\Sigma_{\mathrm{broad}})\) |
| **B1** | Learned mean + isotropic \(\sigma_0^2 I\) |
| **B2** | Learned mean + sensitivity \(\Sigma_{\mathrm{sens},\phi}\) (primary) |
| **B3** | Same \(\mu_\phi\), oracle FD \(h\) covariance (**diagnostic ceiling**) |

Primary efficiency metric (X1, not X0):

\[
AUC_S=\mathrm{AUC}_{\log B}\,S(B),\qquad
B\in\{16,32,64,128,256\}
\]

plus later equal-budget proposal-seeded CEM vs vanilla CEM. Core
scientific contrast is **B2 vs B1** (same \(\mu\), same total variance,
same \(B\)). Optional summary \(B_{80}\) / \(R_B\) vs CEM is X1+, not X0.

X1 failure patterns (pre-declared, not X0 gates):
`mean_only_success`, `sensitivity_prediction_failure`,
`anisotropy_no_value`, `proposal_failure`.

Diffusion / large mixture: **PLAN-X2 only**, and only if unimodal
Gaussian is intrinsically insufficient (`proposal_failure` / explicit
multimodal host).

## X3 note (locked)

\[
\text{sensitivity }\neq\text{ epistemic uncertainty.}
\]

Task curvature may set \(\Sigma_{\mathrm{task}}\propto H_J^{-1}\); execution
covariance must still be gated by world-model validity
\(\Sigma_{\mathrm{exec}}=g(U_{\mathrm{WM}})\Sigma_{\mathrm{task}}\).
R7–R9 validity attaches here, not in X0.

## Claims ceiling (family)

**Allowed after the matching cell PASSes:** amortized proposal reduces
online rollouts under a frozen instrument; H1/H2 isolated by
trace-matched controls.

**Forbidden now:** PLAN-X0 claiming sampling-efficiency or beating CEM;
using P0 failure to unfreeze CEM; mixing PLAN \(R_B\) into CAP \(R_P\);
starting X1/X2/X3 implementation before the prior cell’s prereg freeze
and PASS; unlocking R10.

## Unlock

\[
\text{PLAN-X0 formal PASS}
\;\Rightarrow\;
\text{freeze PLAN-X1 (net, }\epsilon,\,\sigma_0,\,\text{budget, seeds)}
\]

## Ledger

```text
CAP-X0       = PASS
CAP-X1       = PASS
CAP-X2       = COMPLETE  pattern=reference_failure
               R_P(0)=1; R_P(rho>0)=undefined
CAP-X3-P0    = PASS
CAP-X3       = PASS  structured_adaptation_advantage
CAP-X3B      = LOCKED

PLAN-X0      = PASS
PLAN-X1      = FAIL  pattern=anisotropy_no_value
H1           = SUPPORTED
H2           = REJECTED on this host/task
PLAN-X1.5    = FAIL  iso_sufficient
PLAN prior   = μ+iso sufficient on this host (no diffusion)
PLAN-X2      = LOCKED (not triggered)
TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll
TASK-X1      = LOCKED
TASK-XL      = DESIGN FROZEN ONLY; LOCKED
RoboTwin-X0R = FAIL / excitation_failure; G1 PASS
RTWX-X0S     = RAN / FAIL timing_mismatch
RTWX-X0F     = RAN / FAIL force_channel_unresolved
R10          = LOCKED
(PLAN-X: state+goal can learn a search center; TASK-X0 did not show
a hand phase table further improves the action distribution.
Retry = learned progress vs no-progress. TASK-X1 diffusion LOCKED.
Do not rewrite “progress unproven” as “need a stronger generator.”)
```

PLAN-X0 implementation starts only after this freeze on the usual
explicit-start convention for a new cell — **without interrupting
CAP-X2**.
