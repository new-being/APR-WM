# VIS-X3 Preregistration — Perception Uncertainty → Physics-Attribution Veto

Date: 2026-08-17  
Status: **FROZEN**; **VIS-X3 PASS** (`vis_x3_passed=true`;
`outcome_pattern=attribution_success`)
Depends on: `REPORT/REP/VISX/VISX2_REPORT.md` (`vis_x2_passed=true`,
`false_physics_alarm_evidence=true`), frozen X2 alarm
(`W=50`, \(\theta\) from X2 cal; **do not retune**),
`aprwm_v0/vis_x1.py` / `vis_x2.py`  
Does not: redesign / improve the RGB-D estimator; retune X2 \(W/\theta\);
train a new detector; delete residual alarms; claim “physics is fine”
when \(A=1,D=1\); unlock R10-C0; add camera-pose shift

## Why X3 (not better vision)

VIS-X2 closed the causal chain to the diagnosis layer:

\[
\text{perception degradation}
\to E_{\dot q}\uparrow
\to E_{\mathrm{pseudo}}\uparrow
\to P(A=1)\uparrow,
\]

with nominal oracle residual still \(\sim10^{-16}\). False physics alarms
are established.

VIS-X3 must **not** chase a better visual model or retune
\(W=50,\theta\approx0.367\). The clean question is:

\[
\boxed{
\textbf{Can learner-visible perception uncertainty prevent
observation inadequacy from being mis-attributed as model inadequacy?}
}
\]

Family arc:

\[
\boxed{
\begin{array}{ll}
\mathrm{VIS\text{-}X0}:& \text{visual state interface works}\\
\mathrm{VIS\text{-}X1}:& \text{perception error manufactures pseudo-residual}\\
\mathrm{VIS\text{-}X2}:& \text{pseudo-residual triggers false physics diagnosis}\\
\mathrm{VIS\text{-}X3}:& \text{perception uncertainty enables source-aware attribution?}
\end{array}
}
\]

Target thesis fragment:

\[
\boxed{
\textbf{A self-diagnosing world model must distinguish
model inadequacy from observation inadequacy.}
}
\]

## Sole new object: learner-visible \(U_t\)

VIS-X1 localized pollution in the velocity channel. Reuse the existing
local-linear slope estimator. In the estimation window:

\[
\hat q_j=\alpha+\beta(t_j-t)+e_j,
\qquad
\hat{\dot q}=\hat\beta.
\]

Slope standard error (learner-visible):

\[
\boxed{
U_t
=
\operatorname{SE}(\hat{\dot q}_t)
=
\sqrt{
\frac{\hat\sigma_e^{2}}
{\sum_j(t_j-\bar t)^{2}}
}.
}
\]

Properties required:

- learner-visible
- no oracle \(q,\dot q\)
- no \(r_{\mathrm{visual}}\) / \(r_{\mathrm{oracle}}\)
- no \(b_{\mathrm{true}}\) / physics labels
- no trained network
- aligned with the VIS-X1 velocity-error mechanism

Independent channels:

\[
\boxed{
A_t=\text{physics residual alarm (frozen X2)},
\qquad
U_t=\text{perception confidence}.
}
\]

## Semantic rule: veto attribution, not the alarm

Keep X2 alarm unchanged:

\[
A_t=\mathbf{1}[S_t>\theta],
\qquad
\theta\text{ from X2 calibration (reuse; do not retune)}.
\]

On a **nominal-clean** calibration split for uncertainty only, freeze:

\[
\tau_U=Q_{0.99}(U\mid P=\mathrm{nominal},V=\mathrm{clean}).
\]

Define:

\[
D_t=\mathbf{1}[U_t>\tau_U].
\]

Trinary diagnosis (not a binary kill-switch):

\[
\boxed{
\begin{cases}
A=0 &: \text{no physics-invalidity evidence},\\
A=1,\ D=0 &: \text{physics-supported alarm},\\
A=1,\ D=1 &: \text{observation-ambiguous / attribution veto}.
\end{cases}
}
\]

The third case **must not** be read as “physics is OK.” It only means:

\[
\boxed{
\text{current observation quality is insufficient to attribute the
residual to physics}.
}
\]

This avoids the opposite error under `mismatch + degraded`, where true
physics mismatch and perception degradation can coexist.

## Factorial (unchanged from X2)

| Physics | Perception | X3 role |
|---|---|---|
| nominal | clean | uncertainty negative control |
| mismatch | clean | true-physics positive control |
| nominal | degraded | **false-attribution main cell** |
| mismatch | degraded | ambiguity / coexistence diagnostic |

\[
b_{\mathrm{learner}}=0.10,
\qquad
b_{\mathrm{true}}\in\{0.05,0.15\}.
\]

Degraded family = X2 (RGB scale/bias, depth noise/dropout, occlusion).
No camera shift.

## What to score: physics claim, not raw alarm

\[
C_t^{\mathrm{phy}}
=
\mathbf{1}[A_t=1\land U_t\le\tau_U].
\]

Headline false-attribution rate:

\[
\boxed{
FPR_{\mathrm{phyclaim}}^{\mathrm{perc}}
=
P(C^{\mathrm{phy}}=1\mid P=\mathrm{nominal},V=\mathrm{degraded}).
}
\]

X2 baseline (frozen reference, do not recompute \(\theta\)):

\[
FPR_{\mathrm{perc}}^{\mathrm{X2}}=0.0676.
\]

If X3 works, a large share of those events should move to
**observation-ambiguous**, not pretend the residual vanished.

Also report:

\[
TPR_{\mathrm{phyclaim}}
=
P(C^{\mathrm{phy}}=1\mid P=\mathrm{mismatch},V=\mathrm{clean}).
\]

## Gates

### G0 — X2 carryover

Reuse frozen harness; reproduce within sampling tolerance:

\[
FPR_{\mathrm{clean}}\approx0.005,
\qquad
FPR_{\mathrm{perc}}\approx0.068,
\qquad
TPR_{\mathrm{phy}}\approx0.284.
\]

**Forbidden:** recalibrating \(W/\theta\).

### G1 — uncertainty clean specificity

\[
\boxed{
P(D=1\mid\mathrm{nominal,clean})\le0.02.
}
\]

### G2 — perception sensitivity

\[
\boxed{
P(D=1\mid\mathrm{nominal,degraded})
>
P(D=1\mid\mathrm{nominal,clean})
}
\]

and absolute gap

\[
\boxed{>0.20}.
\]

### G3 — false-attribution suppression (main)

Prefer relative bind to X2 baseline:

\[
\boxed{
FPR_{\mathrm{phyclaim}}^{\mathrm{perc}}
\le 0.5\,FPR_{\mathrm{perc}}^{\mathrm{X2}}
=0.0338.
}
\]

(Alternate absolute floor \(<0.02\) is stricter; use the relative rule
as the contracted main gate.)

### G4 — true-physics retention

\[
\boxed{
TPR_{\mathrm{phyclaim}}
\ge 0.8\,TPR_{\mathrm{phy}}^{\mathrm{X2}}
=0.227.
}
\]

Cannot PASS by vetoing everything.

### G5 — no oracle leakage

Hard contract:

\[
U_t
\not\leftarrow
\{q^{\mathrm{oracle}},\dot q^{\mathrm{oracle}},
r_{\mathrm{oracle}},r_{\mathrm{visual}},
b_{\mathrm{true}},\text{physics label}\}.
\]

Only RGB-D estimator internals.

### `mismatch + degraded` is not a forced-classification gate

Coexistence of model + observation inadequacy ⇒ ambiguous
(\(A=1,D=1\)) is **allowed and often correct**. Do not force a binary
choice. Consistent with:

\[
\boxed{
\text{cannot prove physics}
\neq
\text{prove physics is correct}.
}
\]

## Pre-registered outcome patterns

| id | meaning |
|---|---|
| **A `attribution_success`** | high \(U\) under degradation; \(FPR_{\mathrm{phyclaim}}^{\mathrm{perc}}\) drops; clean-mismatch \(TPR_{\mathrm{phyclaim}}\) retained → allowed claim: perception-side uncertainty can prevent over-attribution of observation error to model inadequacy |
| **B `uncertainty_nondiscriminative`** | \(U_{\mathrm{degraded}}\approx U_{\mathrm{clean}}\) → channel has no bite; **do not** train a deeper net to rescue this cell |
| **C `over_veto`** | false claims drop but \(TPR_{\mathrm{phyclaim}}\) collapses → uncertainty only marks hard states unreliable |
| **D `mixed_failure` / coexistence** | heavy ambiguous under mismatch+degraded → **not automatic FAIL**; may be correct |

## Route ledger

```text
VIS-X0 = PASS
VIS-X1 = PASS
VIS-X2 = PASS
VIS-X3 = NEXT
R10    = LOCKED
```

Key semantic freeze: **veto physics attribution; do not veto residual
alarm itself.**
