# VIS-X0 Preregistration — GT Seg + Depth State Interface Smoke

Date: 2026-08-17  
Status: **FROZEN**; **VIS-X0 PASS** (`vis_x0_passed=true`); unlocks **VIS-X1 only**
Depends on: `REPORT/REG/VISX/VISX_PREREG.md`,
`aprwm_v0/simx_plant.py` (`simx_hinge.v1` dynamics)  
Does not: RGB-only; learn a detector; domain randomization; RoboCasa;
claim real perception; unlock R10-C0; change host inertias/damping

## One question

\[
\boxed{
\text{Given simulator GT segmentation + depth on the SIM-X hinge,
can }\hat q\text{ be recovered so the APR-WM / force residual
interface still closes under oracle-matched physics?}
}
\]

This is a **perception-interface smoke**, not a vision benchmark.

## Plant

Use `simx_hinge_vis.v1`: same 1-DoF vertical hinge dynamics as
`simx_hinge.v1` (\(b_0=0.10\), `qfrc_applied` drive, no contact), plus:

- one fixed camera viewing the panel
- panel with distinct visual id (color / geom for seg)
- trivial background + fixed light

Oracle \(q,\dot q\) still logged every step (for gates only).

## Perception (frozen for X0)

Input: \((\mathrm{seg}_{\mathrm{GT}}, D)\) from MuJoCo renderer
(segmentation is **simulator ground truth**).

Output: \(\hat q\) (and optionally \(\hat{\dot q}\) by finite difference
on \(\hat q\), method declared in report).

**Forbidden in X0:** training nets; RGB-only pipelines; object detection.

## Residual path

Frozen learner nominal as in SIM-X (\(b_0=0.10\)). Prefer nominal
matched plant first (VIS-X0-C0-like):

\[
r_{\mathrm{visual}}
=
\text{generalized-force residual using }(\hat q,\hat{\dot q},\ldots)
\]

vs

\[
r_{\mathrm{oracle}}
\text{ from }(q,\dot q).
\]

## Gates

| id | pass |
|---|---|
| **G-phys** | host dynamics still match `simx_hinge.v1` accounting (X0 carryover on oracle stream) |
| **G-geom** | \(\mathrm{RMSE}(\hat q-q)\) below a preregistered floor on a frozen excitation (freeze number in report after a short camera campaign; convention gate, not beauty contest) |
| **G-res** | with GT seg+depth state, \(r_{\mathrm{visual}}\) stays near the oracle residual floor under nominal \(b_{\mathrm{true}}=b_0\) |
| **G-label** | artifacts under `runs/vis_x0/`; `source=simulator`; never `hardware` |

Exact numeric floors for G-geom / G-res (frozen in VIS-X0 report after
first camera campaign):

| gate | floor |
|---|---|
| G-geom | \(\mathrm{RMSE}(\hat q-q)<0.05\) |
| G-res | \(\mathrm{NRMSE}(r_{\mathrm{visual}})<0.10\) with \(r_{\mathrm{visual}}\) using \((\hat q,\hat{\dot q},qacc_{\mathrm{plant}})\) |
| G-phys | oracle \(\mathrm{NRMSE}<10^{-4}\) |

Drive note: VIS-X0 uses PD tracking of bounded joint references so
\(|q|\) stays in the fixed-camera FOV. Oracle state synthesizes \(u\)
only; it is never the perception observation.

## Fail ⇒ interface, not “need RoboCasa”

If G-res fails with GT seg: fix camera→state→residual plumbing before
opening VIS-X1.

## Pass ⇒ unlock VIS-X1 only

VIS-X1 drops oracle segmentation. R10 stays locked.
