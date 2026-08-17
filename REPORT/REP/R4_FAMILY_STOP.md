# R4 Family Stop — Conditional Consequential Information

Date: 2026-08-17  
Status: **STOPPED**  
Does not change: any Door / R3-V7 GO, I0 v2 physics, I1/C0/C1/C2 results  
Does not open: R4-C3, more \(q_0\), stronger probe, CNN, nonlinear \(C\)
fit, compliance retune  
Canonical ledger: this file; companion `REPORT/REP/R4_STAGE_FREEZE.md`

## Four-layer separation (frozen)

\[
\boxed{
\begin{aligned}
I(X;\ell\mid h^{S})&>0 &&\text{local state observable}\\
I(X;e_t^{\mathrm{macro}}\mid h^{S})&\le 0 &&\text{no current-warrant increment}\\
C_{\mathrm{future}}&>0 &&\text{weak future consequence exists}\\
I(X;C_{\mathrm{future}}\mid h^{S})&\le 0 &&\text{no increment on consequence size}
\end{aligned}
}
\]

I1: spatial allocation is identifiable beyond a full macro history.  
C0: that information does not enter B.3 current-residual warrant.  
C1 F0: it produces a \(\sim 10^{-3}\) future-macro fork.  
C2: A/B identity \(\neq\) pair-level \(C\) magnitude; \(L_C(h^{S})=L_C(h^{S},X)=1.055\).

## What this is not

Not “tactile was badly encoded.” Not “the latent is irrelevant.”

\[
\boxed{
\text{observing a latent variable}
\neq
\text{observing the decision-relevant variation of that latent variable}
}
\]

\(X\) answers “stiff-left or stiff-right?” APR-WM needed “given \(h^{S}\),
how much will the future differ?” In this family the second is no.

## Compression principle

\[
\boxed{
\text{representation capacity should follow conditional predictive value,
not physical observability}
}
\]

The leftover principle is stricter than “only model consequential state”:

\[
\boxed{
\textbf{represent variables only when their observable variation carries
conditional information about consequential variation}
}
\]

中文：

> **不是“这个潜变量重要吗”，而是“我当前能观测到的这个潜变量的变化，是否在已有 belief 之外，进一步解释了未来重要结果的变化”。**

That is the same cut as

\[
\text{observing a latent}
\neq
\text{observing its decision-relevant variation}.
\]

## Stage ledger

\[
\boxed{
\begin{aligned}
R4\text{-I0 v1}&:\mu\text{ leaks}\quad\times\\
R4\text{-I0 v2}&:\text{macro-equivalent/local-distinct}\quad\checkmark\\
R4\text{-I1}&:\text{local tactile observability beyond }h^{S}\quad\checkmark\\
R4\text{-C0}&:\text{current warranted value}\quad\times\\
R4\text{-C1 F0}&:\text{weak future consequence exists}\quad\checkmark\\
R4\text{-C1 F1}&:\text{binary target degenerate}\\
R4\text{-C2}&:\text{continuous future consequence increment}\quad\times\\
\textbf{R4 family}&:\textbf{STOP}
\end{aligned}
}
\]

Forbidden reopenings: extra \(C\) samples, probe retune, nonlinear
regression, CNN, stronger tactile, larger \(a^{\mathrm{diag}}\), larger
compliance contrast.

## If a later main stage exists, it is R5 not C3

See `REPORT/REG/R5_PREREG.md` (**not run**). Do not implement it now.
The next real work, if any, is a **physical mechanism**, not model code:
a local DOF that \(X\) sees must continuously control future \(C\) while
current \(h^{S}\) stays approximately indistinguishable.

Program cut:

\[
\boxed{
R3/V7:\text{ conditional modality value}
\rightarrow
R4:\text{ conditional consequential information}
\rightarrow
R5:\text{ consequence-varying hidden state}
}
\]
