# VIS-EXT0 Preregistration — Robosuite Door Visual External-Validity Bridge

Date: 2026-08-17  
Status: **FROZEN**; **PASS** (`vis_ext0_passed=true`;
`outcome_pattern=attribution_external_validity`);
implementation complete (`REPORT/REP/VISX/VISEXT0_REPORT.md`)  
Depends on: `REPORT/REP/VISX/VISX3_REPORT.md` (`attribution_success`),
`REPORT/REG/VISX/VISX_PREREG.md`  
Does not: RoboCasa365 / kitchen task diversity; contact / grasping;
full robot dynamics mismatch; new physics operator; learned uncertainty;
retune APR-WM; unlock R10-C0; claim real perception or sim-to-real

## One question

\[
\boxed{
\text{Does the VIS-X observation-vs-model attribution mechanism
still hold under a more complex visual distribution?}
}
\]

Not a new physics study. Not an estimator beauty contest. Raise **visual
realism only**; keep physics complexity frozen.

VIS-X3 baseline (controlled hinge):

\[
\text{pseudo-residual}
\to
\text{false physics alarm}
\to
\text{perception uncertainty veto},
\]

with \(FPR_{\mathrm{phyclaim}}^{\mathrm{perc}}=0\) and clean-mismatch
\(TPR_{\mathrm{phyclaim}}=0.284\) retained.

## Configuration (frozen)

| item | VIS-EXT0 |
|---|---|
| env | **robosuite `Door` + Panda** |
| physics focus | door hinge **1-DoF only** |
| door drive | generalized-force / controlled trajectory on the hinge |
| Panda | visual clutter / occluder only; **no contact with door** |
| camera | fixed primary camera + preregistered camera perturbations |
| learner input | **RGB-D only** |
| GT segmentation | evaluation / IoU only; **never** into learner |
| estimator | mask + depth → door orientation \(\hat q\) → local-linear \(\hat{\dot q}\) |
| uncertainty | keep \(U_t=\mathrm{SE}(\hat{\dot q})\) |
| residual alarm | keep VIS-X2 **mathematical form** of \(S_t\) |
| attribution | keep \(A=1,U>\tau_U\Rightarrow\) observation-ambiguous |

Robot, table, handle, background, and occlusion enter the image; contact
physics must not contaminate the cell.

### RGB segmentation (no color threshold)

Use a **frozen, zero-finetune** video segmenter (e.g. SAM 2): first-frame
click/box, then propagate masks over the sequence. Official promptable
image/video segmentation; no Door-specific training.

Anti-leakage:

\[
\boxed{
\text{first-frame box/click = fixed human prompt}
}
\]

**not** auto-prompted from simulator GT segmentation. GT instance seg
may be logged for IoU diagnostics only.

## Vision factors only (preregistered)

\[
\xi\in
\{\mathrm{clean},\,\mathrm{texture/light},\,\mathrm{camera},\,
\mathrm{occlusion},\,\mathrm{combined}\}.
\]

Use robosuite camera pose/FOV and texture/material randomization where
available.

**Forbidden this cell:** contact; grasping; full robot dynamics mismatch;
new physics operator; learned uncertainty; RoboCasa task diversity.

## Physics × vision: \(2\times5\)

\[
P\in\{\mathrm{nominal},\,\mathrm{known\ damping\ mismatch}\},
\qquad
V=\xi.
\]

Hard requirement on **all nominal-\(P\)** cells:

\[
\boxed{r_{\mathrm{oracle}}\approx0
\quad(\mathrm{NRMSE}<10^{-4})}.
\]

If oracle accounting breaks, **stop VIS-EXT0** — do not treat later
alarms as a vision result.

## Do not copy numeric \(\theta,\tau_U\) from VIS-X3

Plant, image geometry, and torque scale change. Freeze the **calibration
rule**, not the hinge numbers:

\[
\theta=Q_{0.99}(S\mid P=\mathrm{nominal},V=\mathrm{clean},\mathrm{cal\ split}),
\]

\[
\tau_U=Q_{0.99}(U\mid P=\mathrm{nominal},V=\mathrm{clean},\mathrm{cal\ split}).
\]

Compute once before the formal split and freeze. This is a

\[
\boxed{\textbf{mechanism external-validity bridge}}
\]

not zero-shot cross-scene transfer (that would be a later cell).

## Gates

| id | requirement |
|---|---|
| **G0** Physics closure | \(\mathrm{NRMSE}(r_{\mathrm{oracle}})<10^{-4}\) on nominal-\(P\) |
| **G1** Realistic challenge | on combined and/or occlusion: \(E_{\dot q}^{\mathrm{deg}}>E_{\dot q}^{\mathrm{clean}}\) and \(E_{\mathrm{pseudo}}^{\mathrm{deg}}>E_{\mathrm{pseudo}}^{\mathrm{clean}}\) |
| **G2** Masquerade replication | under nominal \(P\): \(FPR_{\mathrm{alarm}}^{\mathrm{visual}}>FPR_{\mathrm{alarm}}^{\mathrm{clean}}\) |
| **G3** Attribution rescue (main) | \(FPR_{\mathrm{phyclaim}}^{\mathrm{visual}}\le 0.5\,FPR_{\mathrm{alarm}}^{\mathrm{visual}}\) |
| **G4** Physics retention | clean perception + true damping mismatch: \(TPR_{\mathrm{phyclaim}}\ge0.8\,TPR_{\mathrm{alarm}}\) |

## Claims ceiling

**If PASS**, upgrade from “works on controlled synthetic hinge” to:

\[
\boxed{
\textbf{observation-vs-model attribution survives a more realistic,
cluttered robot-manipulation visual distribution}
}
\]

**Still forbidden:** real perception validated; real physics; sim-to-real
solved. Robosuite remains MuJoCo simulation.

## Why not RoboCasa now

Need only:

\[
\text{simple synthetic visual}
\to
\text{realistic cluttered synthetic visual}.
\]

Route:

\[
\boxed{
\textbf{VIS-EXT0: robosuite Door}
\to
\text{if PASS, then decide whether a RoboCasa subset is worth it}
}
\]

## One-line principle

\[
\boxed{\textbf{Raise visual realism; do not simultaneously raise physics complexity.}}
\]

## Route ledger

```text
VIS-X0–X3 = PASS   (controlled hinge arc closed)
VIS-EXT0  = PASS   robosuite Door visual external-validity bridge
RoboCasa  = OPTIONAL next (unlocked by EXT0 PASS; not auto-started)
R10       = LOCKED
```

Implementation complete; see `REPORT/REP/VISX/VISEXT0_REPORT.md`.
