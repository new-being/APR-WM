# VIS-EXT0 Report — Robosuite Door Visual External-Validity Bridge

Date: 2026-08-17  
Status: **`vis_ext0_passed=true`**; **`outcome_pattern=attribution_external_validity`**  
Prereg: `REPORT/REG/VISX/VISEXT0_PREREG.md`  
Depends: VIS-X3 PASS (`attribution_success`)  
Artifacts: `runs/vis_ext0/formal/`  
Does not: unlock R10; claim real perception; claim sim-to-real; open RoboCasa
without an explicit next-cell decision

## Question

\[
\boxed{
\text{Does the VIS-X observation-vs-model attribution mechanism
still hold under a more complex visual distribution?}
}
\]

## Setup

| item | value |
|---|---|
| env | robosuite `Door` + Panda (frozen; no contact drive) |
| physics | hinge 1-DoF Mode-A; learner \(b_0=0.10\); mismatch \(b=0.0\) |
| camera | `agentview` RGB-D; \(\xi\) post-render / camera-pose factors |
| segmenter | `classical_depth_flood_inbox.v1` (SAM2 unavailable); frozen human box/click |
| GT seg | IoU diagnostics only; never into learner |
| alarm | \(W=50\), \(\theta=Q_{0.99}\) on nominal+clean (Door-recalibrated) |
| veto | \(U_t=\mathrm{SE}(\hat{\dot q})\), \(\tau_U=Q_{0.99}\) same cal split |
| claim | \(C^{\mathrm{phy}}=A\land(U\le\tau_U)\) |

Frozen Door thresholds (do **not** copy hinge numbers):

\[
\theta=0.2597,\qquad \tau_U=0.0415.
\]

## Metrics

| metric | value | gate |
|---|---:|---|
| \(\mathrm{NRMSE}(r_{\mathrm{oracle}})\) on nominal-\(P\) | \(<10^{-4}\) | G0 |
| \(E_{\dot q}^{\mathrm{hard}} / E_{\dot q}^{\mathrm{clean}}\) | \(1.25 / 0.084\) | G1 |
| \(E_{\mathrm{pseudo}}^{\mathrm{hard}} / E_{\mathrm{pseudo}}^{\mathrm{clean}}\) | \(0.125 / 0.008\) | G1 |
| \(FPR_{\mathrm{alarm}}^{\mathrm{clean}}\) | 0.0055 | — |
| \(FPR_{\mathrm{alarm}}^{\mathrm{visual}}\) | **0.241** | G2 > clean |
| \(FPR_{\mathrm{phyclaim}}^{\mathrm{visual}}\) | **0.086** | G3 ≤ 0.5×alarm |
| \(TPR_{\mathrm{alarm}}\) (clean mismatch) | 0.300 | — |
| \(TPR_{\mathrm{phyclaim}}\) | **0.284** | G4 ≥ 0.8×alarm |

All of G0–G4 true → **`attribution_external_validity`**.

## Interpretation

On the cluttered Door image distribution, perception \(\xi\) still manufactures
false residual alarms (\(FPR_{\mathrm{alarm}}^{\mathrm{visual}}\gg FPR_{\mathrm{alarm}}^{\mathrm{clean}}\)),
but learner-visible \(U_t\) cuts physics claims by more than half
(\(FPR_{\mathrm{phyclaim}}=0.086\le 0.5\times 0.241\)). Under clean perception +
known damping mismatch, physics claims are largely retained
(\(TPR_{\mathrm{phyclaim}}/TPR_{\mathrm{alarm}}=0.945\)).

Allowed wording:

\[
\boxed{
\textbf{observation-vs-model attribution survives a more realistic,
cluttered robot-manipulation visual distribution}
}
\]

Still forbidden: real perception validated; real physics; sim-to-real solved.
Robosuite remains MuJoCo simulation.

## Segmenter note

SAM 2 was not installed in `.venv-robosuite`. EXT0 uses a frozen human
first-frame box/click + depth-flood propagator under the same anti-leakage
contract (prompt ≠ GT auto-prompt). GT element segmentation is logged for
IoU only.

## Unlock

\[
\boxed{
\text{VIS-EXT0 PASS}\;\Rightarrow\;
\text{RoboCasa subset is now an optional next decision, not locked by EXT0}
}
\]

R10 remains locked. Paper-level real-physics claims remain frozen.
