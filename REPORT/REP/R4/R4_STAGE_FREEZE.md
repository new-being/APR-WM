# R4 Stage Freeze — Local Observability without Macro-Residual Warrant

Date: 2026-08-17  
Status: **FROZEN** (R4 family **stopped** after C2)  
Does not change: I0 physics, I1 \(h^{S}\), `R4_C0_GO=false`, Door V7  
Does not open: tactile encoder, I0 retune, stronger \(a^{\mathrm{diag}}\)  
Canonical stop: `REPORT/REP/R4/R4_FAMILY_STOP.md`. Next family, if any: R5
prereg only (`REPORT/REG/R5/R5_PREREG.md`, not run).

## Frozen chain

\[
\boxed{
\begin{aligned}
R4\text{-I0 v1}&:
\text{scalar }\mu\text{ leaks globally}\quad\times\\
R4\text{-I0 v2}&:
\text{macro-equivalent / locally-distinct}\quad\checkmark\\
R4\text{-I1}&:
I(X;\text{local hidden state}\mid h^{S})>0\quad\checkmark\\
R4\text{-C0}&:
I(X;e_t^{\mathrm{macro-residual}}\mid h^{S})>0\quad\times\\
R4\text{-C1}&:
F0\ \checkmark;\ F1\text{ binary }e\text{ degenerate}\quad\times\\
R4\text{-C2}&:
I(X;C_{\mathrm{future}}\mid h^{S})>0\quad\times
\end{aligned}
}
\]

## New boundary (in addition to Door)

Already known: \(\text{observable}\neq\text{conditionally useful}\).

R4-C0 adds:

\[
\boxed{
\text{locally observable}
\neq
\text{warranted under a macro-residual evidence definition}
}
\]

B.3 \(w_t\) asks whether hidden dynamics have already produced a
**current macro residual consequence**. I0 v2 was designed so they have
not (\(C_{\mathrm{macro}}\sim 6\times10^{-7}\), \(w_t\equiv0\)).
Tactile still sees spatial allocation (I1). The epistemic target, not
the channel, is the bottleneck.

## What C0 does not authorize

- redesigning tactile
- training an encoder
- reconstructing I0 to make B.3 \(w_t\) fire

## Family stop (C2)

Continuous \(C\) has a small but nonzero spread. \(L_C(h^{S})\approx L_C(h^{S},X)\approx 1\).
Tactile does not quantify future-macro magnitude beyond \(h^{S}\).

Do **not** train an encoder. Do **not** retune the probe. See
`REPORT/REP/R4/R4_FAMILY_STOP.md`.
