# VIS-X3 Report — Perception Uncertainty → Physics-Attribution Veto

Date: 2026-08-17  
Status: **`vis_x3_passed=true`**; **`outcome_pattern=attribution_success`**  
Prereg: `REPORT/REG/VISX/VISX3_PREREG.md`  
Depends: VIS-X2 PASS (`θ=0.3668` frozen; **not retuned**)  
Artifacts: `runs/vis_x3/formal/`  
Does not: unlock R10; claim \(A=1,D=1\) means physics is OK

## Question

\[
\boxed{
\text{Can learner-visible }U_t=\mathrm{SE}(\hat{\dot q})
\text{ prevent observation inadequacy from being mis-attributed
as model inadequacy?}
}
\]

## Setup

| item | value |
|---|---|
| \(A_t\) | frozen X2 residual alarm (\(W=50\), \(\theta=0.3668\)) |
| \(U_t\) | slope SE from local-linear fit (no oracle / residual / \(b_{\mathrm{true}}\)) |
| \(\tau_U\) | \(Q_{0.99}(U\mid\mathrm{nominal,clean})=0.0755\) |
| \(C^{\mathrm{phy}}\) | \(A=1\land U\le\tau_U\) |
| factorial | identical to VIS-X2 |

## Metrics

| metric | value | gate |
|---|---:|---|
| \(FPR_{\mathrm{clean}}\) | 0.0053 | G0 |
| \(FPR_{\mathrm{perc}}\) | 0.0676 | G0 |
| \(TPR_{\mathrm{phy}}\) | 0.284 | G0 |
| \(P(D=1\mid\mathrm{nom,clean})\) | **0.005** | G1 ≤0.02 |
| \(P(D=1\mid\mathrm{nom,degraded})\) | **0.924** | G2 gap ≫0.20 |
| \(FPR_{\mathrm{phyclaim}}^{\mathrm{perc}}\) | **0.000** | G3 ≤0.0338 |
| \(TPR_{\mathrm{phyclaim}}\) | **0.284** | G4 ≥0.227 |
| ambiguous (mismatch+degraded) | 0.274 | allowed coexistence |

All of G0–G5 true → **`attribution_success`**.

## Interpretation

Under nominal+degraded, residual alarms still fire at the X2 rate, but
**every** such event is marked observation-ambiguous (\(A=1,D=1\)), so
physics claims drop to zero. Under mismatch+clean, physics claims are
fully retained (\(TPR_{\mathrm{phyclaim}}=TPR_{\mathrm{phy}}\)).

Allowed wording:

\[
\boxed{
\text{perception-side uncertainty can prevent observation error
from being over-attributed to model inadequacy}
}
\]

without deleting the residual alarm itself.

Mismatch+degraded remains largely ambiguous (correct coexistence
behavior; not forced binary classification).

## Family closure (controlled hinge)

\[
\boxed{
\begin{array}{ll}
\mathrm{VIS\text{-}X0}:& \text{visual state interface works}\\
\mathrm{VIS\text{-}X1}:& \text{perception error manufactures pseudo-residual}\\
\mathrm{VIS\text{-}X2}:& \text{pseudo-residual triggers false physics diagnosis}\\
\mathrm{VIS\text{-}X3}:& \text{perception uncertainty enables source-aware attribution}
\end{array}
}
\]

Thesis fragment on this host:

\[
\boxed{
\textbf{A self-diagnosing world model must distinguish
model inadequacy from observation inadequacy.}
}
\]

## Unlock

Next bridge: **VIS-EXT0 PASS** (`REPORT/REP/VISX/VISEXT0_REPORT.md`) —
robosuite Door visual external validity. R10 remains locked. RoboCasa is
an optional next decision (not auto-started).
