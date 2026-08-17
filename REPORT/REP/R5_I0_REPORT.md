# R5-I0 Report — Local Slip-Margin Feasibility (No Neural Probe)

Date: 2026-08-17  
Status: **`R5_I0_PASS=false`** (G_local fail)  
v2: **`R5_I0_V2_PASS=false`** (`REPORT/REP/R5_I0_V2_REPORT.md`)  
Prereg: `REPORT/REG/R5_I0_PREREG.md`  
Artifacts: `runs/r5_i0/formal/summary.json`  
Does not change: R4 STOP  
Does not train: any probe

## Question

Does mean-preserving left/right \(\mu(\lambda)\) give, at hold-end \(t_\star\),

\[
\partial h^{S}/\partial\lambda\approx 0,\quad
\partial X/\partial\lambda\neq 0,\quad
\partial C/\partial\lambda\neq 0
\]

on a \(\lambda\) interval? No learner.

## Result

| Gate | Value | Pass |
|---|---|---|
| G_macro \(\max D(h^{S}_\lambda,h^{S}_0)\) | **0** | true |
| G_local Spearman\((\lambda,X_{\mathrm{L}}-X_{\mathrm{R}})\) | **0** (\(X_{\mathrm{LR}}\equiv 0\)) | false |
| G_cons Spearman\((\lambda,C)\) | **1.0** (\(C\) from \(0\) to \(0.20\)) | true |

\[
\boxed{\texttt{R5\_I0\_PASS}=\text{false}}
\]

\(C(\lambda)\) is ordered: future macro fork grows monotonically with
\(\lambda\). Current \(h^{S}\) is identical across \(\lambda\) (hold,
\(F_t^{\mathrm{cmd}}=0\)). Current tactile shear is also identically
zero, so \(X\) does **not** see the slip-margin coordinate at \(t_\star\).

## Interpretation

This candidate makes a **future** consequence-varying DOF, but that DOF
is not in the **current** tactile snapshot. Admission needs

\[
\frac{\partial X}{\partial\lambda}\neq 0
\quad\text{at the same }t_\star\text{ where}\quad
\frac{\partial h^{S}}{\partial\lambda}\approx 0.
\]

Here \(\partial C/\partial\lambda\neq 0\) and \(\partial h^{S}/\partial\lambda\approx 0\),
yet \(\partial X/\partial\lambda=0\). Moving \(t_\star\) into the probe
would likely break macro overlap (wrench then feels \(\mu\)). That is a
new I0, not a gate patch.

No neural probe. R4 stays stopped. Next physics candidate is v2
self-equilibrated prestress (`REPORT/REP/R5_I0_V2_REPORT.md`), not a CNN.
