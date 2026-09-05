# R1-RS1A Stage Freeze — Epistemic Decision Loop Closed

Date: 2026-08-15  
Status: **STAGE FROZEN**

## One-line verdict

\[
\boxed{
\text{Don't ask whether every mismatch can be detected.
Ask whether it is worth detecting, and if so,
what is the cheapest informative intervention.}
}
\]

## Chain (frozen)

\[
\boxed{
\begin{aligned}
RS1A &: \text{typed inadequacy evidence (support repairs C2)}\\
RS1A.1 &: \text{support fusion cannot rescue weak known signal}\\
RS1A.2 &: \text{known weak signal lives mainly in }D_0=\|r_\perp\|\\
RS1A.3 &: \text{detectability is excitation-limited}\\
RS1A.4 &: \text{misses = tolerate }\cup\text{ consequential probe}\\
RS1A.5 &: \text{probe has optimal positive-VoI intensity }(A^\star=1.5A_0\neq A_{\max})
\end{aligned}
}
\]

## Permanently frozen artifacts

### Detector stack

\[
E_{\mathrm{known}}=D_0,\qquad
E_{\mathrm{unknown}}=\text{support violation (offline)},\qquad
\text{typed channels (no unified scalar fusion)}
\]

### Policy populations

\[
\begin{aligned}
\mathcal T &= \{C < C_{\mathrm{tol}}\} && \text{tolerate}\\
\mathcal P &= \{C \ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=0\} && \text{probe}\\
\mathcal R = \mathcal P_{\mathrm{rev}} &= \{C \ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\} && \text{revise-worthy}
\end{aligned}
\]

### Epistemic budgets (architectural)

\[
\begin{aligned}
B_{\mathrm{compute}} &: \text{neural residual vs physics}\\
B_{\mathrm{epistemic}} &: \text{whether / how much to probe (VoI)}\\
B_{\mathrm{revision}} &: \text{whether to change persistent world model}
\end{aligned}
\]

### VoI mini-rule (RS1A.5)

\[
V(a)=\mathrm{flip}(a)\cdot C-\lambda c(a),\quad
c(A)=(A/A_0)^2,\ \lambda=0.0015
\]

Observed: \(V(1.0)<0,\ V(1.5)>0,\ V(2.0)<0\) on probe set.

## Explicitly closed (do not reopen in RS1B)

- Retuning \(D_0\) / support / \(\gamma\) fusion for C1-L recall
- Treating all misses as failures
- Maximizing IG without decision value
- Studying H32 on all historical “accepted” revisions from raw RS1

## Handoff to RS1B

Intake population **only**:

\[
\boxed{\mathcal P_{\mathrm{rev}}}
\]

Orthogonal question:

\[
\boxed{
\text{Given revision is justified, is the revised vector field
safe over long horizons?}
}
\]

See `REPORT/REG/R1/R1_RS1B_PREREG.md`.
