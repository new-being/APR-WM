# R5-I0-SELFSTRESS Report — Hidden Preload Geometric Stiffness

Date: 2026-08-17  
Status: **`R5_I0_SELFSTRESS_PASS=false`** (G_cons / G_xc)  
Prereg: `REPORT/REG/R5/R5_I0_SELFSTRESS_PREREG.md`  
Artifacts: `runs/r5_i0_selfstress/formal/summary.json`  
Does not change: contact-family PAUSE; R4 STOP; I0 v1–v3  
Does not train: any probe  
Does not retune: \(F^{\mathrm{diag}}\) after seeing \(C\)

## Question

Does a symmetric antagonistic-tendon slider, with preload \(\lambda\)
set only by passive rest length, give on \(\lambda\in\{0,2,\ldots,12\}\)

\[
D(h^{S})\approx 0,\quad
X(\lambda)\text{ ordered},\quad
C(\lambda)\text{ ordered},
\]

with \(\mathrm{range}(C)>0.10\) and Spearman\((X,C)>0.80\)? No learner.
External `ctrl` is identical: hold \(u=0\), frozen \(F^{\mathrm{diag}}=1\).

## Result

| Gate | Value | Pass |
|---|---|---|
| G_macro \(\max D(h^{S})\) | **0** | true |
| G_local Spearman\((\lambda,X)\) | **1.0** (\(X=\lambda\) to numerical noise) | true |
| G_cons Spearman\((\lambda,C)\) | **0.750** | false |
| G_range \(\max C-\min C\) | **1.159** | true |
| G_xc Spearman\((X,C)\) | **0.750** | false |

\[
\boxed{\texttt{R5\_I0\_SELFSTRESS\_PASS}=\text{false}}
\]

At \(t_\star\) (hold end): \(q=v=u=\tau_{\mathrm{motor}}=0\) for every
\(\lambda\); \(T_L=T_R=\lambda\). Hidden rest-length preload does **not**
leak into motor effort. \(h^{S}\) is identical. Local \(X\) is the
intended tension coordinate.

| \(\lambda\) | \(D_h\) | \(X=(T_L+T_R)/2\) | \(C\) | \(q_{\mathrm{end}}\) |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0.0214 |
| 2 | 0 | 2 | 0.501 | 0.0133 |
| 4 | 0 | 4 | 0.920 | 0.0100 |
| 6 | 0 | 6 | 1.119 | 0.0052 |
| 8 | 0 | 8 | 1.159 | 0.0036 |
| 10 | 0 | 10 | 1.137 | 0.0040 |
| 12 | 0 | 12 | 1.115 | 0.0038 |

## Interpretation

The structural claim for **current** macro-null + local visibility
holds exactly:

\[
\partial h^{S}/\partial\lambda=0,\qquad
\partial X/\partial\lambda=1.
\]

Future displacement generally shrinks as \(\lambda\) rises, as
geometric stiffness requires. \(C=\|Y_\lambda-Y_0\|_2\) (relative) is
large and mostly increasing, then **folds** after \(\lambda=8\)
(Spearman \(0.75<0.80\)). Trajectory-level \(C\) is not the same
monotone coordinate as \(X\) on the full interval: late-\(\lambda\)
waveforms differ in more than a static gain, so distance-to-\(Y_0\)
is not ordered even while \(q_{\mathrm{end}}\) stays small.

This is not a contact-pattern accident and not a \(h^{S}\) leak. It is
G_cons/G_xc as registered: \(X\) and \(C\) must be **jointly** ordered,
not three separate coincidences. Do not pass by shrinking the \(\lambda\)
grid, swapping \(C\) to \(|q_{\mathrm{end}}|\) after the fact, or
raising \(F^{\mathrm{diag}}\).

No neural probe. Contact family remains paused.

I0 verdict is frozen. A separate stage, R5-I1-SELFSTRESS, asks whether
\(X\) reduces uncertainty in the full \(Y^{\mathrm{future}}\) given
\(h^{S}\) (`REPORT/REG/R5/R5_I1_SELFSTRESS_PREREG.md`). That stage must
not rewrite \(C\) or drop \(\lambda\).
