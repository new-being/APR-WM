# R5-D0-Preflight Report — Oracle Action Ranking

Date: 2026-08-17  
Status: **`R5_D0_PREFLIGHT_PASS=true`** (unlocks D0 regret only)  
Prereg: `REPORT/REG/R5_D0_PREFLIGHT_PREREG.md`  
Artifacts: `runs/r5_d0_preflight/formal/summary.json`  
Does not change: I0 `PASS=false`; I1 `GO=true`; contact PAUSE  
Does not run: \(P_0\) vs \(P_X\) planner; encoder; new scalar \(C\)

## Question

Under the frozen action set and task loss, does oracle

\[
a^\star(\lambda)=\arg\min_{a\in\mathcal A} J(a,\lambda)
\]

take at least two distinct values on \(\lambda\in\{0,2,\ldots,12\}\)?

\(\mathcal A=\{0.4,1.0,2.5\}\,\mathrm{N}\).  
\(J=(q_T-0.010)^2+10^{-5}\overline{u^2}+10\,\overline{(\max(|q|-0.045,0))^2}\).

## Result

| \(\lambda\) | \(J_{\mathrm{cons}}\) | \(J_{\mathrm{mid}}\) | \(J_{\mathrm{agg}}\) | \(a^\star\) |
|---|---|---|---|---|
| 0 | **1.89e-5** | 1.40e-4 | 1.30e-3 | **cons** |
| 2 | 2.74e-5 | **2.07e-5** | 6.01e-4 | **mid** |
| 4 | 3.67e-5 | **9.95e-6** | 1.92e-4 | **mid** |
| 6 | 6.36e-5 | **3.27e-5** | 6.63e-5 | **mid** |
| 8 | 7.46e-5 | **5.03e-5** | 6.27e-5 | **mid** |
| 10 | 7.21e-5 | **4.59e-5** | 6.22e-5 | **mid** |
| 12 | 7.33e-5 | **4.81e-5** | 6.24e-5 | **mid** |

\[
\boxed{\texttt{R5\_D0\_PREFLIGHT\_PASS}=\text{true}}
\]

\(a^\star\) is not constant: \(\lambda=0\) selects conservative \(0.4\,\mathrm{N}\);
\(\lambda\ge 2\) selects intermediate \(1.0\,\mathrm{N}\). Aggressive
\(2.5\,\mathrm{N}\) is never oracle-optimal under this frozen \(J\).

## Interpretation

Predictive information from I1 can in principle change a decision:
the softest preload overshoots \(y^\star=0.010\) under the I1-scale
force, so the weaker action wins only at \(\lambda=0\).

That is a **ranking crossover**, not a monotone three-action schedule
and not yet \(\mathrm{VoI}(X\mid h^{S})>0\). A planner that cannot see
\(X\) and sees identical \(h^{S}\) must pick **one** action for all
\(\lambda\); oracle does not. Whether using \(X\) reduces **executed
regret** is D0 proper (separate prereg). Do not retune \(\mathcal A\)
or \(J\) to make \(a_{\mathrm{agg}}\) win at high \(\lambda\).

Self-stress family is **not** stopped. No planner in this stage.
