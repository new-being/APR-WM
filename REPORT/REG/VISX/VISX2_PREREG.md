# VIS-X2 Preregistration — Perception → False Physics Diagnosis

Date: 2026-08-17  
Status: **FROZEN**; **VIS-X2 PASS** (`vis_x2_passed=true`;
`false_physics_alarm_evidence=true`); unlocks **VIS-X3 only**
Depends on: `REPORT/REP/VISX/VISX1_REPORT.md` (`vis_x1_passed=true`),
`REPORT/REG/SIMX/SIMX1_*` damping mismatch family,
`aprwm_v0/simx_vis_plant.py`, `aprwm_v0/vis_x1.py` estimator  
Does not: optimize / replace the RGB-D estimator; train a new validity
detector; retune \(W\)/quantile/threshold after seeing formal DR;
perception-uncertainty veto; camera-pose randomization (v1); unlock
R10-C0; open VIS-X3 before X2 closes

## Why X2 (not more vision)

VIS-X1 already showed: oracle physics stays on the numerical floor while
RGB-D state error manufactures \(r_{\mathrm{pseudo}}\), almost entirely
through the \(\hat{\dot q}\) channel. Pose-only substitution ≈ no
pseudo-residual; velocity substitution does.

Therefore VIS-X2 must **not** continue optimizing the visual estimator.
The clean question is:

\[
\boxed{
\textbf{Does perception error alone trigger a residual-based
physics-invalidity alarm?}
}
\]

Scientific chain (upgrade from X1):

\[
\boxed{
\textbf{perception error}
\to
\textbf{state-estimation error}
\to
\textbf{pseudo-residual}
\to
\textbf{false model-inadequacy diagnosis}.
}
\]

## Design: physics × perception factorial

Keep frozen:

- host `simx_hinge_vis.v1`
- RGB-D estimator from VIS-X1 (no retraining / no redesign for DR)
- residual definition from VIS-X0/X1
- PD / reference family unless a declared calibration split needs a
  fixed excitation subset

Factors:

\[
P\in\{\mathrm{nominal},\mathrm{mismatch}\},
\qquad
V\in\{\mathrm{clean},\mathrm{degraded}\}.
\]

| physics | perception | meaning |
|---|---|---|
| nominal | clean | negative control |
| mismatch | clean | true physics-invalidity positive control |
| nominal | degraded | **main cell: perception-only false alarm** |
| mismatch | degraded | interaction diagnostic |

### Physics mismatch (reuse SIM-X1; no new mismatch)

\[
b_{\mathrm{learner}}=0.10,
\qquad
b_{\mathrm{true}}\in\{0.05,0.15\}.
\]

Learner residual always uses frozen \(b_0=0.10\). Plant XML damping is
the only physics change.

### Degraded perception (renderer / observation only)

**Do not change the plant.** Pre-register a small, interpretable
perturbation family, e.g.:

\[
\boxed{
\text{RGB illumination / color shift},
\quad
\text{depth noise / dropout},
\quad
\text{partial occlusion}.
}
\]

**v1 forbids camera-pose shift** (adds calibration semantics).

Critical invariant on every **perception-only** (nominal, degraded) cell:

\[
r_{\mathrm{oracle}}\approx 0
\quad(\mathrm{NRMSE}<10^{-4}).
\]

If this fails, alarms cannot be attributed to perception.

## Frozen residual alarm harness (not a new detector)

No trained validity module. On a **calibration split** of
\((P=\mathrm{nominal},V=\mathrm{clean})\) only, define a window score:

\[
S_t
=
\frac{
\sqrt{\frac1W\sum_{j=t-W+1}^{t} r_{\mathrm{visual},j}^{2}}
}{
\sqrt{\frac1W\sum_{j=t-W+1}^{t} \tau_{j}^{2}}+\epsilon}.
\]

Freeze threshold from calibration only:

\[
\theta=Q_{0.99}(S\mid P=\mathrm{nominal},V=\mathrm{clean}).
\]

Formal test alarm:

\[
\boxed{A_t=\mathbf{1}[S_t>\theta]}.
\]

**Forbidden:** retuning \(W\), the quantile, or \(\theta\) after seeing
formal degraded / mismatch results. Declare \(W,\epsilon\) in code
before the formal campaign (freeze numbers in the report header).

This answers only:

> If diagnosis looks only at the physics residual, how badly can
> perception error fool it?

## Primary metrics

### Clean false-alarm floor

\[
FPR_{\mathrm{clean}}
=
P(A=1\mid P=\mathrm{nominal},V=\mathrm{clean}).
\]

Harness sanity: must stay near the calibration target (order \(1\%\)
for \(Q_{0.99}\) if \(A\) is scored per window / per step as declared).

### Perception false alarm (headline)

\[
\boxed{
FPR_{\mathrm{perc}}
=
P(A=1\mid P=\mathrm{nominal},V=\mathrm{degraded})
}
\]

Pass-as-evidence condition for the scientific claim:

\[
FPR_{\mathrm{perc}}\gg FPR_{\mathrm{clean}}
\quad\text{and}\quad
r_{\mathrm{oracle}}\approx 0
\]

⇒ write:

\[
\boxed{
\text{perception degradation can trigger false physics-invalidity alarms}.
}
\]

### True mismatch recall (positive control)

\[
TPR_{\mathrm{phy}}
=
P(A=1\mid P=\mathrm{mismatch},V=\mathrm{clean}).
\]

Must be clearly above \(FPR_{\mathrm{clean}}\). A never-firing harness
cannot demonstrate perception false alarms.

### Attribution chain (logged, not beauty-contested)

\[
\boxed{
\text{perception degradation}
\to
E_{\dot q}\uparrow
\to
E_{\mathrm{pseudo}}\uparrow
\to
P(A=1)\uparrow.
}
\]

Also retain VIS-X1 leave-one-channel / corr diagnostics where cheap.

## Gates

| id | meaning |
|---|---|
| **G-phys-clean** | on nominal cells (clean and degraded), oracle NRMSE \(<10^{-4}\) |
| **G-harness** | \(FPR_{\mathrm{clean}}\) computed; \(\theta\) frozen from calibration only |
| **G-pos** | \(TPR_{\mathrm{phy}}\) clearly exceeds \(FPR_{\mathrm{clean}}\) (positive control alive) |
| **G-perc** | report \(FPR_{\mathrm{perc}}\) vs \(FPR_{\mathrm{clean}}\) + \(E_{\dot q},E_{\mathrm{pseudo}}\) chain |
| **G-label** | `runs/vis_x2/`; `source=simulator`; never hardware / R10; no post-hoc threshold edit |

Exact numeric floors for “≫” / “clearly exceeds” freeze in the VIS-X2
**report** after the campaign (convention gates), without changing
\(\theta\).

## Explicit prohibition (VIS-X3 territory)

**Do not** add perception-uncertainty veto in X2, e.g.

\[
\Sigma_{\mathrm{perception}}\uparrow
\Rightarrow
\text{veto physics alarm}.
\]

X2 must first answer honestly how wrong residual-only diagnosis can be.
If false physics alarms exist, unlock **VIS-X3**:

\[
\boxed{
\text{perception uncertainty}
\to
\text{error-source attribution / alarm veto}
}
\]

i.e. model inadequacy vs observation inadequacy.

## Route ledger

```text
VIS-X0 = PASS   visual-state interface
VIS-X1 = PASS   perception → pseudo-residual
VIS-X2 = NEXT   pseudo-residual → false physics diagnosis
VIS-X3 = LOCKED attribution / uncertainty mitigation
R10    = LOCKED
```
