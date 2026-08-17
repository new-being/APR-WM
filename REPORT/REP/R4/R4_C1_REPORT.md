# R4-C1 Report — Consequence-Warranted Local Epistemic Value

Date: 2026-08-17  
Status: **F0=true**, **F1 not evaluable**, **`R4_C1_GO=false`**, family **not** stopped  
Prereg: `REPORT/REG/R4/R4_C1_PREREG.md`  
Artifacts: `runs/r4_c1/formal/summary.json`  
Does not change: I0 v2, I1 \(h^{S}\), C0 GO, Door V7  
Does not train: encoder. Labels never from \(X\).

## Question

After a frozen diagnostic action, does a currently macro-hidden local
state produce a **future macro** fork?

\[
C=\frac{\|Y^A-Y^B\|_2}{0.5\|Y^A\|_2+0.5\|Y^B\|_2+\varepsilon},
\qquad
Y=\{q,\dot q,F,M\}.
\]

F0: held-out \(\mathbb{E}[C]>10^{-3}\).  
F1 (only if F0): linear probe \(I(X_{t_\star};e^{\mathrm{future}}\mid h^{S}_{t_\star})\)
with \(e=\mathbf{1}\{C>c_{\min}\}\), no \(e=f(X)\).

\(t_\star=\) last hold frame; \(a^{\mathrm{diag}}\): \(F_t^{\mathrm{cmd}}=4.8\)
for \(0.40\,\mathrm{s}\). Split by \(q_0\).

## F0

| Quantity | Value |
|---|---|
| held \(\mathbb{E}[C]\) | **\(0.001181>10^{-3}\)** |
| train \(\mathbb{E}[C]\) | \(0.001207\) |
| held \(C\) | \(0.00115,\;0.00134,\;0.00089,\;0.00134\) |
| macros matched at \(t_\star\) | **true** |

\[
\boxed{\texttt{R4\_C1\_F0}=\text{true}}
\]

This is a **weak** future fork (barely above the gate), but it is not
nuisance: current \(h^{S,A}\approx h^{S,B}\) and future \(Y\) differs
by about \(10^{-3}\) relative L2. **Not** R4 family stop.

## F1

Binary \(e=\mathbf{1}\{C>c_{\min}\}\) is **saturated on train** (all five
train \(C>c_{\min}\)). Held has a single negative (\(C=8.9\times10^{-4}\)).
A two-class probe cannot be fit without inventing a new \(e_t\).

\[
\boxed{\text{F1 not evaluated (degenerate binary target)}}
\]

\[
\boxed{\texttt{R4\_C1\_GO}=\text{false}}
\]

This is **not** “\(X\) failed to read a varying warrant.” The frozen
indicator has almost no label entropy. Do **not** lower \(c_{\min}\) or
raise \(F_t^{\mathrm{probe}}\) to manufacture class balance.

## Interpretation

F0 establishes a thin instance of

\[
\text{currently hidden but future-consequential state}
\]

under the locked probe. The consequence is small: compliance swap is
not a large slip/failure fork; it is a slight future-macro mismatch.

F1’s binary warrant is the wrong granularity for this \(C\) distribution
(mass just around \(c_{\min}\)). Unlock table row: F0 pass, F1 not GO —
R4-C2 is the next stage (`REPORT/REG/R4/R4_C2_PREREG.md`): continuous
\(I(X;C\mid h^{S})\), same frozen probe, more \(q_0\) only.

C0 remains `GO=false`. Epistemic relevance \(\neq\) current prediction
error; here future consequence exists but is weak and not a balanced
binary target.

## What this does not authorize

- I0 retune to enlarge slip
- \(e_t=f(X)\)
- treating I1 AUROC as F1
- encoder / Door V7E
- declaring nuisance (that required F0 fail)
