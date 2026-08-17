# R6-A0 Preregistration — Decision-Sensitive Error Audit

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REP/R5/R5_D0_REPORT.md`, `REPORT/REP/R5/R5_SELFSTRESS_FAMILY_STOP.md`  
Does not change: R5-D0 `GO=false`; \(\mathcal A\); \(J\); LOO; ridge \(\Pi\)  
Locks: retraining \(\pi_0/\pi_X\); \(X\to a^\star\) classifier; rewriting \(J\);
decision-weighted WM loss as a default next step

## Role

Offline attribution of D0: why \(L_Y\downarrow\) but ranking worsened.
No new hidden state. No new planner. Diagnostic gates only, not a
beauty contest.

## Frozen objects

Reuse D0's LOO ridge maps, counterfactual \(Y_a(\lambda)\), \(\hat Y_a\),
\(\mathcal A\), and \(J\). Reconstruct predictions; do not refit a
different estimator.

## Core quantities

\[
e_a^J=\hat J_a-J_a,\qquad
m_b=J_b-J_{a^\star},\qquad
\hat m_b=m_b+(e_b^J-e_{a^\star}^J),
\]

\[
\rho_b=\frac{e_b^J-e_{a^\star}^J}{m_b}.
\]

\(\rho_b<-1\Rightarrow\) that pair's ranking flips. Also

\[
e_a^J\approx g_a^\top\delta Y_a,\qquad
g_a=\nabla_Y J(Y_a),
\]

with \(\delta Y^\parallel\) along \(g_a\) and \(\delta Y^\perp\) the rest.
\(g\) is the analytic gradient of the frozen \(J\) (position, effort,
limit terms).

## Diagnostic gates

- **D1 (attribution):** every D0 ranking error of \(\pi_X\) (and
  \(\pi_0\)) satisfies \(\hat m_{\hat a}<0\) vs \(a^\star\); every
  preserved ranking has \(\hat m_b>0\) for all \(b\neq a^\star\).
- **D2 (metric mismatch):** mean trajectory \(L_Y\) (physical MSE,
  average over \(a\in\mathcal A\)) is lower for \(\pi_X\) than
  \(\pi_0\), **and** \(\pi_X\) has more \(a^\star\)-pair margin
  violations (\(\hat m_b<0\)) or larger mean \(|e_b^J-e_{a^\star}^J|\)
  on the nearest competitor.
- **D3 (linear explanation):** report \(R^2\) of \(g^\top\delta Y\) vs
  \(e^J\) over all \((\lambda,a,\pi)\). No pass threshold; \(R^2<0.5\)
  flags strongly nonlinear \(J\).

A0 does not set R6 GO. **R6-A1** audits pairwise \(\hat m\) uncertainty
in \(J\)-margin space (nested LOO), not a delta-method on \(g\).
