# R4-I0 Preregistration — Contact-Regime Closure

Date: 2026-08-17  
Status: **FROZEN** (protocol **v2** after v1 `PASS=false`; v2 not a C0 GO)  
Depends on: Door / R3-V7 closed; v1 report `REPORT/REP/R4/R4_I0_REPORT.md`  
Does not change: `REPORT/REG/R4/R4_C0_PREREG.md` scientific question  
Locks: R4-C0 formal, encoder training, RGB, Door V7E; **scalar \(\mu\) as the hidden label**

## Role

Same class as R1-MJ0 / R1-RS2-C0: **causal-preflight**, not belief GO.

v1 hid the label in \(\mu_{\mathrm{C0}}\neq\mu_{\mathrm{C1}}\) and leaked
through \(F_n\). That no-go is retained. v2 asks for a many-to-one map:

\[
\boxed{
X_{\mathrm{local\ contact}}
\longrightarrow
(F_n,F_t,q,\dot q)
}
\]

## Frozen protocol v2

\[
\mu^{A}=\mu^{B},\qquad F_n^{\mathrm{preload},A}\approx F_n^{\mathrm{preload},B}.
\]

Hidden variable: left/right **contact compliance swap** (MuJoCo `solref`
as a \(k_t/k_n\) spatial proxy), equal area so

\[
\bar k^{A}=\bar k^{B}.
\]

Pattern A: stiff left / soft right. Pattern B: the swap.

Same cuboid, same squeeze, same \(F_t(t)\) command. No RGB, no Door.

Tactile: \(X=\{p,\tau_x,\tau_y\}\).

## Minimal I0 question (v2)

\[
\boxed{
\exists t:\
h_t^{S,A}\approx h_t^{S,B}
\ \land\
X_t^{A}\not\approx X_t^{B}
}
\]

Ordered stick→incipient→gross is **not** required for this round.

## Gates (no classifier)

**Gate 0 — global mechanics matched.** Count frames with

\[
|q^A-q^B|<\varepsilon_q,\
|\dot q^A-\dot q^B|<\varepsilon_v,\
\frac{|F_n^A-F_n^B|}{\bar F_n}<\varepsilon_n,\
\frac{|F_t^A-F_t^B|}{\bar F_t}<\varepsilon_t
\]

and likewise motor/command overlap (controller must not carry the label).
Need \(N_{\mathrm{matched}}\ge 8\). If \(N=0\), fail as in v1.

**Gate 1 — local tactile differs on those frames.**

\[
D_{\mathrm{taxel}}=\|\tau^A-\tau^B\|_2
\quad\text{and/or}\quad
D(p^A,p^B),\
\Delta\text{shear centroid}.
\]

Need \(N_{\mathrm{counterexample}}\ge 8\) with \(D_{\mathrm{taxel}}>D_{\min}\).

Target:

\[
D(h^{S,A},h^{S,B})\approx 0,\qquad D(X^A,X^B)\gg 0.
\]

No AUROC, no \(e_t\), no GRU. Failure ⇒ the pad model may lack local DOF;
do not train a network.

## Unlock

`R4_I0_PASS=true` (v2 gates) unlocks **R4-I1** only. I1 still checks
\(p(h^S\mid A)\approx p(h^S\mid B)\) including \(a_t,\tau_{\mathrm{motor}},F_{\mathrm{wrist}}\).
