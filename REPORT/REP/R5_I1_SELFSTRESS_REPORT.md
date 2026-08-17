# R5-I1-SELFSTRESS Report — Conditional Future-Response Information

Date: 2026-08-17  
Status: **`R5_I1_SELFSTRESS_GO=true`**  
Prereg: `REPORT/REG/R5_I1_SELFSTRESS_PREREG.md`  
Artifacts: `runs/r5_i1_selfstress/formal/summary.json`  
Does not change: **`R5_I0_SELFSTRESS_PASS=false`**; contact PAUSE; R4 STOP  
Does not train: encoder / Adam / decision scalar \(C\)

## Question

With I0 physics frozen and \(h^{S}\) identical at \(t_\star\), does
local preload \(X\) reduce held-out risk on the full future macro
trajectory \(Y=\{q,\dot q,\ddot q,\tau_{\mathrm{motor}}\}\)?

\[
\Delta_Y=L_Y(P_0)-L_Y(P_X),\qquad
P_0:h^{S}\mapsto Y,\quad
P_X:(h^{S},X)\mapsto Y.
\]

Ridge only. Train \(\lambda\in\{0,4,8,12\}\), held \(\{2,6,10\}\).

## Result

| Gate | Value | Pass |
|---|---|---|
| H_var \(L_Y(P_0)\) | **0.545** | true |
| H_gain \(\Delta_Y/L_Y(P_0)\) | **0.387** | true |
| H_shuffle gap vs shuffled \(X\) | **0.387 − (−1.046)** | true |

\[
\boxed{\texttt{R5\_I1\_SELFSTRESS\_GO}=\text{true}}
\]

\(L_Y(P_X)=0.335\). Shuffled \(X\) **raises** held loss (\(L=1.116\)).
Self-stress family is **not** stopped.

## Interpretation

I0 remains a valid no-go on scalar \(C=\|Y-Y_0\|_2\): that radial
coordinate folds. I1 shows the underlying map \(\lambda\mapsto Y_\lambda\)
is still predictable from the locally observed preload once \(h^{S}\)
is conditioned on.

This is the split I0 made visible:

\[
I(X;\|Y-Y_0\|\mid h^{S})
\quad\text{need not be a monotone coordinate,}
\]

while

\[
I(X;Y^{\mathrm{future}}\mid h^{S})>0
\]

in the registered predictive-risk sense.

That is **not** a decision-consequence definition, not an encoder GO,
and not a rewrite of I0. Next, if any, is how to read a decision
functional off \(Y^{\mathrm{future}}\) **after** this admission — not
another hidden-state search, and not shopping \(|q_{\mathrm{end}}|\)
back into I0.
