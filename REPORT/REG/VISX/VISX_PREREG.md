# VIS-X Preregistration — Visual Observation Bridge on SIM-X Host

Date: 2026-08-17  
Status: **FROZEN** (family); **VIS-X0–X3 PASS**; **VIS-EXT0 PASS**
(`REPORT/REP/VISX/VISEXT0_REPORT.md`); RoboCasa optional next;
**R10 locked**;
does not unlock R10-C0; does not reopen SIM-X3 / REAL-LOG-A1  
Depends on: `REPORT/REP/SIMX/SIMX_FAMILY_STOP.md`,
`REPORT/REP/PAPER/APRWM_R7_R9_SIMX_SYNTHESIS.md`,
`aprwm_v0/simx_plant.py` / `simx_vis_plant.py`  
Does not: ManiSkill/SAPIEN assets; RoboCasa; RLBench migration as
first cell; download large manipulation corpora; claim real vision;
claim perception solved; train end-to-end policies

## Why VIS-X (not another data lake)

REAL-LOG-A0 showed public logs are not force-auditable. The cheap next
controlled question is **not** “find a bigger dataset,” but:

\[
\boxed{
x_t^{\mathrm{visual}}
\to
\hat s_t
\to
\text{frozen APR-WM / force residual}
}
\]

on the **same** mechanical host already closed in SIM-X.

## Host discipline

Physics plant stays the SIM-X hinge family:

\[
\texttt{simx\_hinge.v1}
\quad\text{(dynamics)}
\qquad
\texttt{simx\_hinge\_vis.v1}
\quad\text{(same dynamics + camera / texture)}
\]

`simx_hinge_vis.v1` must **not** change \(I\), damping nominal \(b_0\),
integrator, or drive convention. Only add:

- fixed camera
- visually marked hinge panel (color/texture)
- simple background + illumination

Each physics step logs:

\[
(x_t^{\mathrm{RGB}},\,x_t^{\mathrm{depth}},\,[\mathrm{seg}],\,
q_t^{\mathrm{oracle}},\,\dot q_t^{\mathrm{oracle}}).
\]

Generate on the fly:

\[
\boxed{\mathrm{simulate}\to\mathrm{render}\to\mathrm{save\ paired\ visual+oracle}}
\]

## Observation swap (controlled)

**Before (SIM-X):** \(s_t^{\mathrm{oracle}}\to\) APR-WM.  
**VIS-X:** \(s_t^{\mathrm{oracle}}\to\mathrm{renderer}\to x_t\to\hat s_t\to\) APR-WM.

Sole new factor:

\[
\boxed{\textbf{perception}}
\]

## Cells (in order)

1. **VIS-X0 — GT segmentation + depth smoke** — **PASS**
   (`REPORT/REP/VISX/VISX0_REPORT.md`).
   \((\mathrm{seg},D)\to\hat q\). No detection/texture learning.
   Interface smoke only. Simulator seg ≠ real perception.

2. **VIS-X1 — RGB-D** — **PASS** (`REPORT/REP/VISX/VISX1_REPORT.md`)
   Sole new variable: **drop GT segmentation**. Same host/camera/RGB-D/
   PD/residual/oracle accounting. Headline:

   \[
   \boxed{r_{\mathrm{pseudo}}=r_{\mathrm{visual}}-r_{\mathrm{oracle}}}
   \]

   Evidence: oracle floor intact; \(E_{\mathrm{pseudo}}\) rises; velocity
   channel dominates attribution → perception can masquerade as model
   inadequacy.

3. **VIS-X2 — false physics diagnosis** — **PASS**
   (`REPORT/REP/VISX/VISX2_REPORT.md`). \(FPR_{\mathrm{perc}}=0.068\)
   vs \(FPR_{\mathrm{clean}}=0.005\) with \(r_{\mathrm{oracle}}\approx0\);
   \(TPR_{\mathrm{phy}}=0.28\); chain holds.

5. **VIS-EXT0 — robosuite Door visual bridge** — **PASS**
   (`REPORT/REP/VISX/VISEXT0_REPORT.md`). Attribution external validity
   under cluttered Door RGB-D: \(FPR_{\mathrm{phyclaim}}^{\mathrm{visual}}=0.086
   \le 0.5\,FPR_{\mathrm{alarm}}^{\mathrm{visual}}\); clean-mismatch
   \(TPR_{\mathrm{phyclaim}}\) retained. Raise vision only; no contact/grasp;
   frozen human prompt (SAM2 unavailable → classical depth-flood substitute);
   Door-recalibrated \(\theta,\tau_U\).

Later (only if an explicit RoboCasa cell is opened): RoboCasa subset.
**Not auto-started.**

## Claims ceiling

VIS-X may support:

- perception→state→APR-WM **interface** closure (X0)
- pseudo-residual from perception error (X1)
- false physics-invalidity alarms from perception-only degradation (X2)
- source-aware attribution via learner-visible perception uncertainty (X3)
- attribution external validity on cluttered Door RGB-D (EXT0)

VIS-X may **not** claim: real-world vision; R10-C0; SIM-X3 lifecycle
rescue; “solved perception”; that \(r_{\mathrm{visual}}\neq0\) is
physics mismatch; that \(A=1,D=1\) means physics is OK; sim-to-real.

## Route

\[
\boxed{
\begin{array}{c}
\texttt{simx\_hinge.v1}\\
\downarrow\\
\text{MuJoCo camera render}\\
\downarrow\\
(RGB,D,\mathrm{seg})\\
\downarrow\\
\hat q,\hat{\dot q}\\
\downarrow\\
\text{frozen APR-WM / residual}\\
\downarrow\\
\text{VIS-EXT0: robosuite Door}
\end{array}
}
\]

## Locked ledger

```text
VIS-X0–X3 = PASS   controlled hinge arc closed
VIS-EXT0  = PASS   Door visual external-validity bridge
RoboCasa  = OPTIONAL next (not auto-started)
R10       = LOCKED
```

Next decision: whether a RoboCasa subset is worth it.
Principle: raise visual realism; do not raise physics complexity yet.
