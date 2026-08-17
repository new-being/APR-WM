# R1-RS1A.2 Preregistration — Weak Structural Signal Detection

Date: 2026-08-15  
Status: **FROZEN** — formal completed; see `REPORT/REP/R1/R1_RS1A2_REPORT.md` (`RS1A.2_GO=false`; \(D_0\) still strongest for C1-L)

## Placement

\[
\boxed{
\begin{aligned}
RS1 &\rightarrow RS1A \rightarrow RS1A.1 \rightarrow RS1A.2\\
&\rightarrow \text{(only then)}\; RS1B \rightarrow RS1C
\end{aligned}
}
\]

Converged split from RS1A / RS1A.1:

\[
\boxed{
\text{C2 = support-evidence problem},\quad
\text{C1-L = weak-signal detection problem}
}
\]

**Do not** continue \(\gamma\) / support-composition for C1-L.  
**Do not** open RS1B.  
**Do not** run revision / acceptance / passivity / H32.

## Scientific question

\[
\boxed{
\text{Why does a weak but persistent in-library structural residual
fail to separate from adequate physics?}
}
\]

More operationally: which statistic reveals C1-L against C0 —

\[
\text{magnitude consistency}
\quad vs\quad
\text{direction consistency}
\quad vs\quad
\text{temporal correlation}
\quad vs\quad
\text{operator-aligned matched evidence}
\]

Hypothesis to test (not assume):

> Weak model inadequacy is easier to detect in operator-aligned space
> than in residual-energy space.

## Frozen exclusions

- no \(\gamma\) sweep
- no support fusion into the C1-L score
- no MuJoCo-specific operators added to the library
- support remains a **separate** unknown-structure channel

## Residual statistics (discovery window; same sampling as RS1A)

Let \(r_t=r_{\perp,t}\) (scalar hinge residual after tangent projection).

| ID | Formula | Hypothesis |
|----|---------|------------|
| \(D_0\) | \(\mathrm{RMS}(r)\) | magnitude (strongest current baseline) |
| \(D_{\mathrm{SNR}}\) | \(\mathbb E[|r|]/\sqrt{\mathrm{Var}(|r|)+\epsilon}\) | weak but stable amplitude |
| \(D_{\mathrm{dir}}\) | \(\bigl\|\frac1T\sum_t r_t/(|r_t|+\epsilon)\bigr\|\) | consistent sign/direction |
| \(D_{\mathrm{corr}}\) | \(\frac1{T-1}\sum_t \frac{r_tr_{t+1}}{|r_t||r_{t+1}|+\epsilon}\) | temporal coherence ≠ \(\sum z^2\) |
| \(D_{\mathrm{lib}}\) | \(\max_{k\in\mathcal L} \frac{\bigl|\sum_t r_t\phi_k(x_t,v_t)\bigr|}{\sqrt{\sum_t\phi_k^2}+\epsilon}\) | library-matched alignment |

\(\mathcal L\) = frozen V4/R0.6 operator library (includes `abs_v_v`, but detector
does **not** receive the true regime label or true \(\alpha\)).

Secondary (architecture, not C1-L competitor):

\[
D_{\mathrm{support}}=S(u)=-\log(1-u+\epsilon)
\]

from the frozen RS1A support audit.

## Typed epistemic channels (secondary endpoint)

\[
\boxed{
\begin{cases}
D_{\mathrm{known}} := D_{\mathrm{lib}}\\
D_{\mathrm{unknown}} := D_{\mathrm{support}}
\end{cases}
}
\qquad
\mathrm{trigger}
=
(D_{\mathrm{known}}>\tau_k)
\;\lor\;
(D_{\mathrm{unknown}}>\tau_u)
\]

Baseline typed control uses \(D_{\mathrm{known}}:=D_0\) with the same OR rule.

Threshold rule (frozen):

- \(\tau_u\): smallest value such that C0 support-trigger rate \(=0\) on the
  evaluation set’s C0 scores (equivalently \(\tau_u=\epsilon\) when C0 has \(u=0\))
- \(\tau_k\): C0 FPR \(\le 1\%\) operating point on the known channel alone
  (same construction as RS1A/A.1)

Because C0/C1 have \(u=0\), the OR rule does not inflate C0/C1 false triggers
via support; C2 is covered by the unknown channel.

## Regimes / labels

| Regime | Primary use |
|--------|-------------|
| C0 | negative |
| C1-L | **primary positive** (weak in-library) |
| C1-H | secondary positive (strong in-library) |
| C2-latch | typed unknown-channel diagnostic only |
| CNEG | diagnostic only |

## Matrix

New held-out seeds (not used in RS1A / RS1A.1):

\[
\{9501,9511,9521,9531,9541\}
\times
\{C0,C1\text{-}L,C1\text{-}H,C2\text{-}latch\}
\times 8\ \mathrm{probes}
=
160
\]

+ CNEG ×40 diagnostic. Smoke: `8971` ×5 ×P1.

## Primary endpoints

For each residual statistic \(D\in\{D_0,D_{\mathrm{SNR}},D_{\mathrm{dir}},D_{\mathrm{corr}},D_{\mathrm{lib}}\}\):

1. AUROC / AUPRC on **C0 vs C1-L**
2. Operating point at C0 FPR \(\le 1\%\): C1-L recall (and report C1-H)

### Hard GO (weak-signal success)

\[
\boxed{
\exists\, D\in\{D_{\mathrm{SNR}},D_{\mathrm{dir}},D_{\mathrm{corr}},D_{\mathrm{lib}}\}
\text{ s.t. }
R_{\mathrm{C1\text{-}L}}(D)>R_{\mathrm{C1\text{-}L}}(D_0)
\text{ at C0 FPR}\le1\%
}
\]

and

\[
\boxed{
AUROC_{\mathrm{C1\text{-}L}}(D)
\ge
AUROC_{\mathrm{C1\text{-}L}}(D_0)-\delta
},\qquad\delta=0.02
\]

If multiple qualify, the **primary winner** is the one maximizing C1-L recall at
that operating point (tie-break: higher AUROC).

### Typed secondary (report; not required for GO)

At the frozen OR rule:

- C1-L recall of \((D_{\mathrm{lib}}\lor D_{\mathrm{support}})\) vs \((D_0\lor D_{\mathrm{support}})\)
- C2 recall of both typed systems
- C0 false-trigger ≤1%

## Interpretation guide

| Outcome | Meaning | Next |
|---------|---------|------|
| \(D_{\mathrm{lib}}\) wins | weak inadequacy → operator-aligned space | freeze typed channels; open RS1B drafting |
| coherence (\(D_{\mathrm{dir}}\)/\(D_{\mathrm{corr}}\)) wins | persistence = consistent structure, not \(\sum z^2\) | redesign detector; still not RS1B until C1-L gate holds |
| nothing beats \(D_0\) | information may be in sampling/excitation, not statistic | probe/design study before RS1B |
| \(D_{\mathrm{SNR}}\) alone wins | weak stable amplitude | keep magnitude family, improve SNR estimator |

## Explicit non-goals

- beating pooled AUROC by sacrificing C1-L
- reintroducing support into the C1-L scalar
- unlocking RS2
