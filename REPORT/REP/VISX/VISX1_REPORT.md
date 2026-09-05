# VIS-X1 Report — RGB-D Perception Pseudo-Residual

Date: 2026-08-17  
Status: **`vis_x1_passed=true`** / **`vis_x1_contract_ok=true`**; unlocks **VIS-X2 only**  
Prereg: `REPORT/REG/VISX/VISX1_PREREG.md`  
Depends: VIS-X0 PASS (`runs/vis_x0/formal/summary.json`)  
Artifacts: `runs/vis_x1/formal/`  
Does not: claim real vision; unlock R10-C0; introduce validity/certificate

## Question

\[
\boxed{
\text{Does dropping GT segmentation (RGB-D only) manufacture a
physics-like pseudo-residual on the frozen SIM-X hinge?}
}
\]

Sole new variable vs VIS-X0: **no GT seg in the estimator**.

## Setup (frozen with X0)

| item | value |
|---|---|
| host / camera / PD / residual | identical to VIS-X0 |
| perception | RGB color mask → depth unproject → pivot PCA → local-linear \(\hat{\dot q}\) |
| RGB gate | \(R>1.5G+1\), \(R>1.5B+1\), \(R>60\) (classical; not learned) |
| cells | 3×3 × 4 s @ 2 ms |

## Headline metrics

| metric | VIS-X1 (max / mean) | VIS-X0 baseline |
|---|---|---|
| \(E_q\) (max) | **0.050** | 0.049 |
| \(E_{\dot q}\) (max) | **1.69** | (piecewise high in X0 too) |
| \(E_r=\mathrm{NRMSE}(r_{\mathrm{visual}})\) (max) | **0.032** | 0.031 |
| \(E_{\mathrm{pseudo}}=\mathrm{RMS}(r_{\mathrm{pseudo}})\) (max) | **0.169** | mean cell pseudo \(\approx0.057\) |
| oracle NRMSE (max) | \(\sim10^{-16}\) | \(\sim10^{-16}\) |

Gates: **G-phys / G-run / G-contrast / G-attrib / G-label** all true.

## Attribution

| quantity | value |
|---|---|
| \(\mathrm{corr}(\|r_{\mathrm{pseudo}}\|,\|\hat{\dot q}-\dot q\|)\) (mean) | **1.00** |
| \(\mathrm{corr}(\|r_{\mathrm{pseudo}}\|,\|\hat q-q\|)\) (mean) | 0.47 |
| \(\mathrm{RMS}\) pseudo with visual-\(q\) + oracle-\(\dot q\) | \(\sim10^{-15}\) |
| \(\mathrm{RMS}\) pseudo with oracle-\(q\) + visual-\(\dot q\) | **0.058** |
| dominant channel | **velocity** |

Leave-one-channel replay: pose-only substitution does **not** open a
pseudo-residual; velocity-channel substitution does. On this plant and
residual definition, perception error enters mainly through
\(\hat{\dot q}\).

## Scientific reading

\[
r_{\mathrm{oracle}}\approx 0
\quad\text{but}\quad
E_{\mathrm{pseudo}}\ \text{and}\ E_r\ \text{are nonzero}
\]

⇒

\[
\boxed{
\text{perception error can masquerade as model inadequacy}.
}
\]

**Never** read \(r_{\mathrm{visual}}\neq 0\) alone as physics mismatch.
`perception_masquerade_evidence=true` in `summary.json`.

Piecewise cells dominate \(E_{\mathrm{pseudo}}\) (discontinuous refs →
harder \(\hat{\dot q}\)); sine/chirp stay milder (\(E_{\mathrm{pseudo}}\sim0.01\text{–}0.02\)).

## Unlock

**VIS-X2** may open (`REPORT/REG/VISX/VISX2_PREREG.md`): residual-only
false physics-invalidity alarms under perception degradation.  
**VIS-X3** (uncertainty veto) stays locked. R10 remains locked.
