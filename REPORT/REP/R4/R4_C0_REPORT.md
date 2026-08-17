# R4-C0 Report — Observation Necessity of Tactile given \(h^{S}\)

Date: 2026-08-17  
Status: **GO=false** (`R4_C0_GO=false`)  
Prereg: `REPORT/REG/R4/R4_C0_PREREG.md`  
Artifacts: `runs/r4_c0/formal/summary.json`  
Smoke: `runs/r4_c0/smoke/summary.json` (not GO)  
Depends on: frozen I0 v2, I1 PASS  
Does not change: I0 \(\mu\)/`solref`/grid/squeeze/\(F_t(t)\), I1 \(h^{S}\), Door V7  
Does not train: tactile encoder / CNN

## Question (unchanged)

\[
I(X_{\mathrm{tactile}};e_t\mid h^{S})>0\ ?
\]

\[
P_0:h^{S}\to e_t
\qquad\text{vs}\qquad
P_X:(h^{S},X)\to e_t,
\qquad
\Delta_X=L(P_0)-L(P_X),\ \varepsilon=0.005.
\]

\(e_t=y\,w_t\) is the B.3 recipe transferred to this domain, not Door
weights. \(y=0\) on stiff-left (A), \(y=1\) on stiff-right (B). \(w_t\)
is cumulative squared divergence of **noiseless macro** residuals
(I1 \(h^{S}\) channels, no taxels). If relative macro energy
\(<(10^{-4})\), warrant is null: integrating numerical dust must not
turn I1’s A/B observability into a fake \(e_t\).

GRU32, raw flattened \(X\) concatenated (linear GRU input; **no CNN**).
Split by unseen \(q_0\). I0 physics frozen.

## Result

| Quantity | Value |
|---|---|
| \(L(h^{S})\) | \(0.239\) |
| \(L(h^{S},X)\) | \(0.254\) |
| \(\Delta_X\) | \(-0.015\le\varepsilon\) |
| shuffle \(\Delta\) | \(0.029\) (shuffle better than aligned \(X\)) |
| held relative macro energy | \(6.1\times 10^{-7}\) |
| held warrant active | **0** |
| held mean \(w_t\) on B | **0** |

\[
\boxed{\texttt{R4\_C0\_GO}=\text{false}}
\]

Smoke plumbing only; GO not evaluated there.

## Interpretation

I0/I1 already showed

\[
I(X;\text{hidden local state}\mid h^{S})>0.
\]

C0 asked whether that local state is **warranted structural evidence**.
Under the transferred recipe it is not: A/B macros do not diverge, so
\(w_t\equiv 0\) and \(e_t\equiv 0\). Adding \(X\) does not reduce loss
and slightly worsens it — the same qualitative pattern as Door C.5/V7D
(extra modality as unused/noisy channels).

\[
\boxed{
I(X;\text{hidden local state}\mid h^{S})>0
\not\Rightarrow
I(X;e_t\mid h^{S})>0
}
\]

Tactile sees a proprioceptively hidden spatial allocation. That
allocation does not generate the B.3 evidence process on this mixture.
I1’s AUROC \(0.996\) was **observability**, not epistemic value.

Door vs R4 (this operational test):

\[
\begin{aligned}
\text{Door}:&\quad I(Z;e_t\mid h^{S})\le 0\\
\text{R4}:&\quad I(X;e_t\mid h^{S})\le 0
\end{aligned}
\]

Modality value remains distribution- and belief-conditional. This
domain made local contact *visible* given \(h^{S}\); it did not make it
*necessary for* \(e_t\).

## What this does not authorize

- retuning I0/I1
- training a tactile encoder because \(\Delta_X\le\varepsilon\)
- treating I1 AUROC as C0 GO
- Door V7E / RGB fusion / larger GRU

## Next

Do **not** train an encoder. Do **not** patch \(w_t\) or I0.
Ledger: `REPORT/REP/R4/R4_STAGE_FREEZE.md`.
Next, if any: **R4-C1** (`REPORT/REG/R4/R4_C1_PREREG.md`) — future-macro
consequence warrant, labels never from \(X\). Not run.
