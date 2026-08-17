# R6-B0 Preregistration — Frozen Uncertainty-Abstention Realization

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R6/R6_A1_REPORT.md`, frozen R5-D0  
Does not change: \(\mathcal A\); \(J\); \(z_{\min}=1\); nested-LOO \(\sigma_{ab}\)  
Locks: threshold sweep; \(\lambda=0\) special case; LCB redesign;
\(J\)-weighted WM; new classifier on this family

## Role

Not a planner-optimization stage. One frozen abstention rule, then
stop this realization branch.

Question: does \(X\)'s proposed **deviation from** \(\pi_0\) have a
1-RMS certificate? Split

\[
\text{harmful-switch suppression}
\quad\text{vs}\quad
\text{useful-switch realization}.
\]

A1 already predicts collapse: \(z_{\mathrm{cons,mid}}<1\) at every
\(\lambda\), so \(\pi_B=\pi_0\) identically.

## Frozen rule

Reuse A1's \(z_{ab}\) and \(z_{\min}=1\). Let \(\pi_0,\pi_X\) be D0
actions. If \(\pi_X(\lambda)=\pi_0(\lambda)\), then \(\pi_B=\pi_0\).
If they differ, switch only when

\[
z_{\pi_0(\lambda),\pi_X(\lambda)}\ge 1;
\]

else \(\pi_B=\pi_0\). No other pairs enter the rule.

## Policy-identity preflight

Apply the rule to A1 tables. If \(\pi_B(\lambda)=\pi_0(\lambda)\) for
all \(\lambda\), set

\[
\texttt{B0\_POLICY\_COLLAPSE}=\text{true}
\]

and close **algebraically**: \(\bar R_B=\bar R_0\) from D0. Do not
rerun the simulator. \(\pi_B\in\{\pi_0,\pi_X\}\) by construction, so
even a non-collapse case is a D0 regret lookup, not a new rollout.

## Reported quantities

- \(N_{\mathrm{harmful\ proposed}}\), \(N_{\mathrm{harmful\ retracted}}\)
- \(N_{\mathrm{useful\ proposed}}\), \(N_{\mathrm{useful\ accepted}}\)
- \(\bar R_B\), \(\bar R_0\), \(\bar R_X\)

A collapse with \(\bar R_B=\bar R_0\) is **not** “uncertainty-aware
planner success.” Write:

\[
\text{uncertainty repaired harmful realization by abstaining,
but did not recover positive }\mathrm{VoI}_\Pi.
\]

## After B0

This realization branch **stops**. Do not try \(z_{\min}\in\{0.8,0.5,0.2\}\).
Next scientific object is **R7** (`REPORT/REG/R7/R7_PREREG.md`): directional
switch certification on a **new** family. No R6-C0. Ledger:
`REPORT/REP/R6/R6_REALIZATION_FREEZE.md`.
