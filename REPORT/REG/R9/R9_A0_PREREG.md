# R9-A0 Preregistration — Persistent Certificate-Validity Belief

Date: 2026-08-17  
Status: **RAN**; **`R9_A0_GO=true`**; B0 unlocked, not run  
Depends on: `REPORT/REP/R9/R9_P0_REPORT.md` (`PASS=true`)  
Does not: validity decay; change detector; re-probe; online conformal
recalibration; \(F_{\max}\) estimator; R7 \(f_{\mathrm{WM}}\)/\(q\) update;
fuse \(b(\mathcal V)\) with \(L_{\mathrm{WM}}\); Adam

## Question

Can one calibration observation establish a **persistent** belief that
the frozen R7 certificate is currently applicable, and license it for
\(K\) block-stationary tasks?

\[
\boxed{
\text{calibrate}
\to S_{\mathrm{epi}}
\to b(\mathcal V)
\to b\ge\tau_{\mathcal V}
\to L_{\mathrm{WM}}>\delta
\to \mathrm{commit\ cons.}
}
\]

Not: distinguish \(1.5\) vs \(2.0\) vs \(2.2\) as a 3-way label.
\(\mathcal V=1\) iff the frozen certificate remains decision-safe
(ID and benign). Invalidating shift \(F_{\max}=1.5\) has \(\mathcal V=0\).

## Persistence

\[
F_{\max}\text{ is constant on a block of }K\text{ fresh episodes.}
\]

\[
b_{k+1}=b_k.
\]

No decay. Change detection is R9-B0, not A0.

## Frozen objects

P0 probe, R0 PD, B0 \(f_{\mathrm{WM}},q\), \(\delta=10^{-3}\),
\(\varepsilon_{\mathrm{reset}}=2.2\times10^{-3}\).
\(K_{\min}^{\mathrm{oracle}}=2\) from P0; **not retuned**.

Validity-fit \(\alpha\): B0 WM-fit \(\cup\) certificate-cal, **restricted
to \(L_{\mathrm{WM}}>\delta\)** (the only place a license is used; P0
G1 was this locus). Held G-validity: B0 held-test commit-set
\(\times\{1.5,2.0,2.2\}\).

## Belief (one logistic, not a GRU)

\[
\tilde S = S_{\mathrm{epi}}/10^{-3},\qquad
b=\sigma(\beta_0+\beta_1\tilde S),\qquad
\tau_{\mathcal V}=0.5
\]

Scale \(10^{-3}\) is the frozen G1 gap unit, not a held-set fit.
MLE on validity-fit commit-set only, ridge \(10^{-8}\).
Labels are class \(\mathcal V\), not \(A\) or \(J\).

## Two layers (not one score)

\[
\pi=
\begin{cases}
a_{\mathrm{cons}} & b\ge\tau_{\mathcal V}\ \mathrm{and}\ L_{\mathrm{WM}}>\delta,\\
a_{\mathrm{default}} & \mathrm{otherwise.}
\end{cases}
\]

## Block protocol (held, commit-set primary)

For each held \((\alpha,F_{\max})\) with \(L_{\mathrm{WM}}>\delta\):
calibrate once (probe+reset), freeze \(b\), then \(K\) **fresh**
rest-to-task episodes. \(C_{\mathrm{cal}}\) uses the P0 formula and
P0’s \(V_{\mathrm{dec}}\) with that block’s \(T_{\mathrm{cal}}\).
Low-\(\alpha\) (\(L\le\delta\)) never needs a license; they are not in
the NetVoI gate.

Report \(K\in\{1,2,4,8\}\). \(K\) is not a tuning variable.
**Primary NetVoI is \(K=2\).**

Comparators: no-validity B0 (\(C_{\mathrm{cal}}=0\), always license);
persistent \(b\); oracle \(\mathcal V\); shuffled \(S_{\mathrm{epi}}\)
(seed 0) among commit-set blocks.

## Gates (all required for GO)

- **G-validity** held commit-set \(\times\) class: AUROC\(\ge 0.80\),
  Brier\(\le 0.20\)
- **G-safe** two-layer commits: \(P(A>0\mid\mathrm{commit})=1\)
- **G-use** valid blocks (ID+benign): useful recall \(\ge 0.25\)
- **G-persist** episode \(k=2\) commit mask equals \(k=1\) (fresh
  episodes, \(b\) frozen)
- **G-NetVoI** mean commit-set \(\mathrm{NetVoI}(K=2)>0\)

\[
\mathrm{NetVoI}(K)=\frac1N\sum_i\Bigl(K\cdot A_i\cdot\mathbf 1_{\mathrm{commit},i}-C_{\mathrm{cal},i}\Bigr).
\]

If \(K=2\) fails but a larger listed \(K\) is positive: **GO false**,
record \(K_{\min}^{\mathrm{realized}}\) and
\(\Delta K=K_{\min}^{\mathrm{realized}}-2\). Do not edit P0.

## After GO

Unlock R9-B0 (mid-block \(F_{\max}\) change). Not this stage.
