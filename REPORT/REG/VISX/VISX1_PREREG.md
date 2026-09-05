# VIS-X1 Preregistration — RGB-D Perception Pseudo-Residual

Date: 2026-08-17  
Status: **FROZEN**; **VIS-X1 PASS** (`vis_x1_passed=true`); unlocks **VIS-X2 only**
Depends on: `REPORT/REP/VISX/VISX0_REPORT.md` (`vis_x0_passed=true`),
`REPORT/REG/VISX/VISX_PREREG.md`, `aprwm_v0/simx_vis_plant.py`  
Does not: GT segmentation; RGB-only; domain randomization; learned
detectors as the scientific claim; validity/certificate; change
dynamics; unlock R10-C0; open VIS-X2 before X1 closes

## Sole new variable

VIS-X0 already closed the interface with GT seg + depth. VIS-X1 adds
exactly one change:

\[
\boxed{\textbf{drop GT segmentation}}
\]

Everything else stays frozen:

| frozen | same as VIS-X0 |
|---|---|
| host | `simx_hinge_vis.v1` |
| camera | `fixed_cam` pose / intrinsics / resolution |
| RGB-D | MuJoCo renderer RGB + depth |
| drive | PD / reference family (sine, chirp, piecewise) |
| residual definition | \(r(\hat q,\hat{\dot q},qacc_{\mathrm{plant}},\tau;b_0)\) |
| oracle physics accounting | \(r_{\mathrm{oracle}}\) on matched \(b_0\) |
| dynamics | \(I\), \(b_0\), integrator, `qfrc_applied` |
| validity / certificate | **not introduced** |

Pipeline:

\[
\boxed{
RGBD
\to
\hat q,\hat{\dot q}
\to
r_{\mathrm{visual}}
}
\]

## One scientific question

Not “can RGB-D estimate \(q\)?” as a vision contest. The value is:

\[
\boxed{
\text{Does real perception error manufacture a physics-like
pseudo-residual?}
}
\]

Primary diagnostic:

\[
\boxed{
r_{\mathrm{pseudo}}
=
r_{\mathrm{visual}}-r_{\mathrm{oracle}}
}
\]

## Perception (frozen for X1)

**Input:** \((RGB, D)\) only. GT seg is audit-only if rendered; it must
**not** enter the estimator.

**Output:** \(\hat q,\hat{\dot q}\) by a declared classical RGB-D
geometry method (color/depth panel mask → same pivot-PCA / slope
family as X0, or an equally simple non-learned substitute). Method
must be fixed in code before the formal campaign.

**Forbidden in X1:** deep nets as the claim; RGB-only; camera/light/texture
randomization (that is VIS-X2); claiming real-world vision.

## Frozen metrics (all logged every cell)

\[
E_q=\mathrm{RMSE}(\hat q-q),
\qquad
E_{\dot q}=\mathrm{RMSE}(\hat{\dot q}-\dot q),
\qquad
E_r=\mathrm{NRMSE}(r_{\mathrm{visual}}),
\]

and the headline quantity:

\[
\boxed{
E_{\mathrm{pseudo}}
=
\mathrm{RMS}(r_{\mathrm{visual}}-r_{\mathrm{oracle}})
=
\mathrm{RMS}(r_{\mathrm{pseudo}}).
}
\]

### Attribution gate

\[
\mathrm{corr}\big(|r_{\mathrm{pseudo}}|,\,|\hat{\dot q}-\dot q|\big)
\]

plus a declared \(q\) vs \(\dot q\) error decomposition (e.g. also
\(\mathrm{corr}(|r_{\mathrm{pseudo}}|,|\hat q-q|)\) and/or leave-one-channel
replays: visual-\(q\)+oracle-\(\dot q\) vs oracle-\(q\)+visual-\(\dot q\)).
Purpose: decide whether pseudo-residual is mainly pose- or
velocity-driven — **not** to “fix” perception.

## Gates

| id | meaning |
|---|---|
| **G-phys** | oracle accounting still on the VIS-X0 floor (\(\mathrm{NRMSE}<10^{-4}\)) |
| **G-run** | RGB-D estimator produces finite \(\hat q,\hat{\dot q}\) on all cells; no GT seg in the path |
| **G-contrast** | report \(E_q,E_{\dot q},E_r,E_{\mathrm{pseudo}}\) vs VIS-X0 baselines on the same host/drive |
| **G-attrib** | attribution stats logged (corr / decomposition); no pass/fail beauty contest |
| **G-label** | `runs/vis_x1/`; `source=simulator`; `real_perception=false`; never hardware / R10 |

VIS-X1 is **not** required to beat VIS-X0 on \(E_q\). A scientifically
useful outcome is:

\[
r_{\mathrm{oracle}}\approx 0
\quad\text{but}\quad
E_r\ \text{and/or}\ E_{\mathrm{pseudo}}\ \text{rise}
\]

interpreted only as:

\[
\boxed{
\text{perception error can masquerade as model inadequacy}.
}
\]

**Never** interpret \(r_{\mathrm{visual}}\neq 0\) alone as physics mismatch.

Exact numeric floors for any optional soft thresholds (if used) freeze
in the VIS-X1 **report** after the campaign; the scientific deliverable
is the pseudo-residual quantification, not a vision leaderboard.

## Fail / pass semantics

- **Fail G-phys / G-run / G-label:** fix plumbing or labeling; do not
  open VIS-X2.
- **High \(E_{\mathrm{pseudo}}\) with closed oracle:** **PASS as evidence**,
  not a reason to abandon the host — it unlocks the VIS-X2 question.
- **Low \(E_{\mathrm{pseudo}}\) (RGB-D ≈ GT-seg):** also informative;
  still report metrics; VIS-X2 may need stronger appearance stress.

## Unlock

Pass (contract complete + artifacts) unlocks **VIS-X2 only**
(`REPORT/REG/VISX/VISX2_PREREG.md`: false physics diagnosis).  
**VIS-X3** (uncertainty veto) stays locked until X2 shows false alarms.  
R10 stays locked. No validity module in X1.

## Route ledger

```text
VIS-X0 = PASS
VIS-X1 = next
VIS-X2 = locked pending X1
R10    = locked
```
