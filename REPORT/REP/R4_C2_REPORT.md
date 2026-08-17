# R4-C2 Report — Continuous Future-Consequence Epistemic Value

Date: 2026-08-17  
Status: **`R4_C2_GO=false`**, **R4 family stop**  
Prereg: `REPORT/REG/R4_C2_PREREG.md`  
Artifacts: `runs/r4_c2/formal/summary.json`  
Does not change: I0 v2, \(a^{\mathrm{diag}}\), \(H\), \(t_\star\), C1 F0, C0  
Does not train: encoder. \(C\) never uses \(X\). A/B share one pair \(C\).

## Question

\[
I(X_{t_\star};C\mid h^{S}_{t_\star})>0\ ?
\]

Ridge linear \(P_0:h^{S}\to z(C)\) vs \(P_X:(h^{S},X)\to z(C)\).
More \(q_0\) only (17 train / 8 held). \(\Delta_C>0.05\) for GO.

## Result

| Quantity | Value |
|---|---|
| train \(\mathbb{E}[C]\pm\mathrm{std}\) | \(0.001215\pm 0.000133\) |
| held \(\mathbb{E}[C]\pm\mathrm{std}\) | \(0.001209\pm 0.000154\) |
| macros matched at \(t_\star\) | true |
| \(L_C(h^{S})\) | \(1.055\) |
| \(L_C(h^{S},X)\) | \(1.055\) |
| \(\Delta_C\) | \(-4.7\times 10^{-8}\) |
| canon \(\Delta\) | \(\approx 0\) |
| shuffle \(\Delta\) | \(0.090\) (not GO; shuffle *better*) |

\[
\boxed{\texttt{R4\_C2\_GO}=\text{false}}
\]

\[
\boxed{\text{R4 family stop}}
\]

\(L_C(P_0)\approx 1\): even \(h^{S}\) does not beat predicting the train
mean of \(z(C)\). Adding raw \(X\) does not change held MSE at numerical
noise. Shuffle looking slightly better is a high-variance artifact on
a narrow \(C\) band, not a tactile channel.

## Interpretation

C1 F0 already established a weak future fork. C2 asks whether current
tactile **quantifies** that fork given \(h^{S}\). It does not.

\[
\boxed{
\begin{aligned}
I(X;\text{local state}\mid h^{S})&>0\\
I(X;e^{\mathrm{macro-residual}}_t\mid h^{S})&\le 0\\
I(X;C_{\mathrm{future}}\mid h^{S})&\le 0
\end{aligned}
}
\]

\[
\boxed{
\text{future consequence exists}
\not\Rightarrow
\text{current tactile helps quantify it beyond }h^{S}
}
\]

Pair-shared \(C\) blocked the I1 A/B shortcut: the probe had to predict
\(0.0009\) vs \(0.0013\), not left vs right. It could not.

## Stop

Do **not** train an encoder. Do **not** retune \(\mu\)/`solref`/\(a^{\mathrm{diag}}\).
Do **not** lower \(c_{\min}\). Ledger: `REPORT/REP/R4_FAMILY_STOP.md`.
If a later main stage exists it is **R5**, not C3
(`REPORT/REG/R5_PREREG.md`, not run).
