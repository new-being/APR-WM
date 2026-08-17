# R5-I1-SELFSTRESS Preregistration — Conditional Future-Response Information

Date: 2026-08-17  
Status: **FROZEN**  
Depends on: `REPORT/REG/R5/R5_I0_SELFSTRESS_PREREG.md`,
`REPORT/REP/R5/R5_I0_SELFSTRESS_REPORT.md`  
Does not change: `R5_I0_SELFSTRESS_PASS=false`; contact-family PAUSE;
I0 \(\lambda\) grid; \(F^{\mathrm{diag}}\); \(C=\|Y-Y_0\|_2\)  
Locks: neural net / Adam; putting \(X\) in \(Y\); rewriting I0 \(C\) as
\(|q_{\mathrm{end}}|\); dropping \(\lambda=10,12\)

## Why this stage exists

I0 showed current-side structure is essentially exact
(\(D_h=0\), \(X=\lambda\)), while scalar

\[
C=\|Y_\lambda-Y_0\|_2
\]

folds after \(\lambda\approx 8\). That is a **scalarization** failure,
not a missing hidden state. I1 asks the world-model question directly:

\[
\boxed{I(X_{t_\star};Y^{\mathrm{future}}\mid h^{S}_{t_\star},a^{\mathrm{diag}})>0\ ?}
\]

## Locked physics

Same sandbox as I0: rest-length preload, hold \(u=0\), frozen
\(F^{\mathrm{diag}}=1\), \(\lambda\in\{0,2,\ldots,12\}\). \(t_\star=\)
last hold frame.

\(h^{S}=(q,v,a,u,\tau_{\mathrm{motor}},u^{\mathrm{cmd}})\) at \(t_\star\).
\(X=(T_L+T_R)/2\). \(a^{\mathrm{diag}}\) is the frozen probe (identical
across rows).

\(Y^{\mathrm{future}}\) is the post-\(t_\star\) slider macro trajectory

\[
\{q,\dot q,\ddot q,\tau_{\mathrm{motor}}\}
\]

(1-DoF; no independent \(M\)). **No** tendon tension in \(Y\).

## Estimator (not a world model)

Ridge linear maps, closed form, no Adam.

\[
P_0:(h^{S})\mapsto Y,\qquad
P_X:(h^{S},X)\mapsto Y.
\]

Features z-scored on train; each \(Y\) coordinate z-scored on train;
loss is mean squared error on that z-scored vector.

**Train \(\lambda\):** \(\{0,4,8,12\}\). **Held \(\lambda\):**
\(\{2,6,10\}\) (interpolation, not endpoint deletion of I0).

Shuffle control: apply \(P_X\) on held with \(X\) permuted.

## Gates

- **H_var:** \(L_Y(P_0)>10^{-6}\) on held (future actually varies)
- **H_gain:** \(\Delta_Y/L_Y(P_0)>0.05\) where
  \(\Delta_Y=L_Y(P_0)-L_Y(P_X)\)
- **H_shuffle:** \(\Delta_Y/L_Y(P_0)\) exceeds the shuffled-\(X\)
  relative gain by at least \(0.05\)

\[
\texttt{R5\_I1\_SELFSTRESS\_GO}=H_{\mathrm{var}}\land H_{\mathrm{gain}}\land H_{\mathrm{shuffle}}.
\]

If GO is false (\(\Delta_Y\le\varepsilon\) in the registered sense),
**stop the self-stress family**. If GO is true, that is physical
admission that \(X\) predicts future macros given \(h^{S}\); it does
**not** reopen I0, does not define a decision scalar, and does not
train an encoder.
