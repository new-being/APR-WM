# R5-D0-Preflight Preregistration — Oracle Action Ranking vs Hidden Preload

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REG/R5_I1_SELFSTRESS_PREREG.md`,
`REPORT/REP/R5_I1_SELFSTRESS_REPORT.md`  
Does not change: `R5_I0_SELFSTRESS_PASS=false`; `R5_I1_SELFSTRESS_GO`;
contact PAUSE; I0 \(C\); \(F^{\mathrm{diag}}\) of I0/I1  
Locks: neural planner; scalar shopping (\(|q_{\mathrm{end}}|\) as new I0
\(C\); RMSE; peak force as a replacement \(C\)); retuning \(\mathcal A\)
or \(J\) after seeing this run

## Role

Oracle **decision-feasibility** only. No \(P_0\)/\(P_X\) planner yet.

I1 admitted predictive information. D0-preflight asks whether that
information can change an action ranking under a **frozen** task loss.
If not, stop the self-stress family at

\[
\text{predictive information}
\neq
\text{decision information}.
\]

## Frozen action set

Same hold (\(u=0\), \(0.30\,\mathrm{s}\)) as I0. Probe window still
\(0.40\,\mathrm{s}\). Actions differ only by a constant lateral force
in that window (not duration shopping):

\[
\mathcal A=\{\,a_{\mathrm{cons}}=0.4,\; a_{\mathrm{mid}}=1.0,\; a_{\mathrm{agg}}=2.5\,\}
\quad\text{(newtons).}
\]

\(a_{\mathrm{mid}}\) equals the I1 diagnostic force; the other two are
fixed conservative / aggressive flanks, chosen before this run.

## Frozen task loss

Positioning a 1-DoF slider to a setpoint, with effort and a joint-limit
barrier — a task loss, not a statistic designed to sort \(\lambda\):

\[
J(Y)
=
\bigl(q_T-y^\star\bigr)^2
+
10^{-5}\,\overline{u^2}_{\mathrm{probe}}
+
10\,\overline{\bigl(\max(|q|-0.045,0)\bigr)^2}.
\]

\(y^\star=0.010\). \(\lambda\) grid unchanged: \(\{0,2,\ldots,12\}\).

## Oracle and gate

For each \(\lambda\), rollout every \(a\in\mathcal A\) in the true
simulator.

\[
a^\star(\lambda)=\arg\min_a J(a,\lambda)
\]

(ties: all \(a\) within \(10^{-12}\) of the min count as the same
rank cell; a unique argmin is required to score a change).

**G_flip:** at least two distinct unique \(a^\star\) occur on the
\(\lambda\) grid.

\[
\texttt{R5\_D0\_PREFLIGHT\_PASS}=G_{\mathrm{flip}}.
\]

If this fails, **stop the self-stress family**. Do not retune
\(\mathcal A\), \(y^\star\), or \(\beta\) to manufacture a crossover.
Do not open the \(h^{S}\)-vs-\((h^{S},X)\) regret comparison.

If it passes, that only unlocks D0 regret (separate prereg). It does
not train a net and does not rewrite I0.
