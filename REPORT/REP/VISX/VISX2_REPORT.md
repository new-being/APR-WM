# VIS-X2 Report — Perception → False Physics Diagnosis

Date: 2026-08-17  
Status: **`vis_x2_passed=true`**; **`false_physics_alarm_evidence=true`**;
unlocks **VIS-X3 only**  
Prereg: `REPORT/REG/VISX/VISX2_PREREG.md`  
Depends: VIS-X1 PASS  
Artifacts: `runs/vis_x2/formal/`  
Does not: retune alarm post-hoc; perception-uncertainty veto; unlock R10

## Question

\[
\boxed{
\text{Does perception error alone trigger a residual-based
physics-invalidity alarm?}
}
\]

## Setup

| item | frozen value |
|---|---|
| host / RGB-D estimator / residual | VIS-X1 (unchanged) |
| factorial | \(P\in\{\mathrm{nominal},\mathrm{mismatch}\}\times V\in\{\mathrm{clean},\mathrm{degraded}\}\) |
| mismatch | \(b_{\mathrm{learner}}=0.10\), \(b_{\mathrm{true}}\in\{0.05,0.15\}\) |
| degraded | RGB scale/bias + depth noise/dropout + occlusion (no camera pose) |
| alarm | \(W=50\), \(\epsilon=10^{-8}\), \(\theta=Q_{0.99}(S\mid\mathrm{nominal,clean})\) |
| cal | seed 9101 / sine / nominal / clean |
| test | seeds \{9111,9121\} × \{sine,chirp\} × full factorial |

\(\theta\) frozen at **0.367** from calibration only (see `alarm_harness.json`).

## Headline rates

| metric | value |
|---|---:|
| \(FPR_{\mathrm{clean}}\) | **0.0053** |
| \(FPR_{\mathrm{perc}}\) | **0.0676** |
| \(TPR_{\mathrm{phy}}\) | **0.284** |
| \(TPR\) (mismatch+degraded) | 0.283 |

Gates: G-phys-clean / G-run / G-harness / G-pos / G-perc / G-label all true.

On all nominal cells (clean and degraded): oracle NRMSE \(\sim10^{-16}\).

## Attribution chain

| step | nominal clean → degraded |
|---|---|
| \(E_{\dot q}\) | 0.116 → **0.217** |
| \(E_{\mathrm{pseudo}}\) | 0.012 → **0.022** |
| \(P(A=1)\) | 0.005 → **0.068** |
| `chain_holds` | **true** |

## Claim (allowed wording)

\[
FPR_{\mathrm{perc}}\gg FPR_{\mathrm{clean}}
\quad\text{and}\quad
r_{\mathrm{oracle}}\approx 0
\]

⇒

\[
\boxed{
\text{perception degradation can trigger false physics-invalidity alarms}.
}
\]

Chain form:

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

**Never** interpret a residual alarm alone as physics mismatch when
oracle accounting remains closed.

## Unlock

**VIS-X3** may open (`REPORT/REG/VISX/VISX3_PREREG.md`): learner-visible
perception uncertainty → veto **physics attribution** (not the residual
alarm). R10 remains locked. Do not retune \(W/\theta\) from these numbers.
