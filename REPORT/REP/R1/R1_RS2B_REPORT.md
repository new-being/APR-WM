# R1-RS2B Report — Contact-Mediated Consequence Transport

Date: 2026-08-16  
Prereg: `REPORT/REG/R1/R1_RS2B_PREREG.md`  
Depends on: `REPORT/REP/R1/R1_RS2_REPORT.md`, `REPORT/REP/R1/R1_RS2A_REPORT.md`  
Artifacts: `runs/r1_rs2b/formal/`  
Packages: `robosuite==1.5.2`, `mujoco==3.11.0`  
Prereg SHA256: `d27f96a7141a3fcaf7f9879c0037a811e47a897c05f5b942b3f61638c132dbd6`

## Decision

\[
\boxed{RS2B\_GO=\mathrm{true}}
\]

\[
\boxed{
\text{Mode-A }C\text{ does not retain decision meaning for
contact-mediated out-of-library harm}
}
\]

This does **not** set `RS2_GO=true`. It does **not** rewrite `RS2A_GO` or
`RS1C_GO`. It does **not** retune \(C_{\mathrm{tol}}\). Detectability stays
closed in RS2A.

Smoke (`runs/r1_rs2b/smoke/`) is plumbing only.

## Question (unchanged)

\[
\boxed{
\text{Does Mode-A consequence }C
\text{ retain its decision meaning under contact-mediated interaction?}
}
\]

C1 detectability was not measured. The split remains:

\[
\boxed{RS2A:\ \text{Can I see the mismatch?}}
\qquad
\boxed{RS2B:\ \text{If I see it, does “how much it matters” still transport?}}
\]

## Matrix

20 cells: Formal seeds \(\times\{\mathrm{C0},\mathrm{C1\text{-}L},\mathrm{C1\text{-}H},\mathrm{C2}\}\).
Frozen `fast_pull` OSC tape on C0; \(H_{32}=32\); eight intervention ICs
(same generator family as Mode-A \(C\)) plus a protocol IC \((0.12,0)\).

\(C_{\mathrm{tol}}=0.001934\) is the frozen Mode-A ruler.

## Gates

| Hypothesis | Result | Detail |
|---|---|---|
| H0 C0 \(\Delta C\) sanity | **PASS** | 0/5 false consequential on either ruler |
| H2a C2 \(C_{\mathrm{ModeA}}<C_{\mathrm{tol}}\) | **PASS** | 5/5 |
| H2b median \(L_{\mathrm{contact}}^{\mathrm{C2}}\ge C_{\mathrm{tol}}\) | **PASS** | \(0.141\gg 0.00193\) |
| H2c \(L_{\mathrm{C2}}>L_{\mathrm{C0}}\) | **PASS** | \(0.141\) vs \(3.3\times10^{-5}\) |

\[
\boxed{RS2B\_GO=H0\land H2=\mathrm{true}}
\]

H1 (in-library \(\Delta C\)) is **reported, not GO**:

| H1 | Result | Detail |
|----|--------|--------|
| H1a decision agreement | pass | \(0.80\) (8/10) |
| H1b Spearman \((C_{\mathrm{ModeA}},C_{\mathrm{contact}})\) | **fail** | \(-0.115\) |

## C2: saw it, judged it unimportant, but it mattered

| seed | \(C_{\mathrm{ModeA}}\) | \(C_{\mathrm{contact}}\) | \(L_{\mathrm{contact}}\) | protocol \(L\) |
|------|----------------------:|-------------------------:|-------------------------:|---------------:|
| 11101 | \(-0.00526\) | \(0.00313\) | \(0.141\) | \(0.177\) |
| 11111 | \(0.00033\) | \(0.00057\) | \(0.097\) | \(0.177\) |
| 11121 | \(0.00108\) | \(-0.00029\) | \(0.159\) | \(0.177\) |
| 11131 | \(\approx0\) | \(-0.00023\) | \(0.125\) | \(0.177\) |
| 11141 | \(0.00089\) | \(-0.00879\) | \(0.152\) | \(0.177\) |

Median \(C_{\mathrm{contact}}^{\mathrm{C2}}=-0.00023\approx0\): oracle-drag
is the incumbent on C2, so the **same functional** as Mode-A \(C\) still
cannot see latch. Raw incumbent loss \(L_{\mathrm{contact}}\) is two to four
orders of magnitude above C0.

That is exactly Formal’s C2 pattern, now isolated from detectability:
`detect=1` was already true; the frozen policy tolerated because
\(C_{\mathrm{ModeA}}<C_{\mathrm{tol}}\); under contact the incumbent is a
bad forecast of latch truth.

## C1: \(\Delta C\) decision bits mostly agree, ranks do not

The two disagreements are the weak C1-L seeds Formal already had below
\(C_{\mathrm{tol}}\) (11101, 11121). Contact \(\Delta C\) is above
\(C_{\mathrm{tol}}\) for both. C1-H agrees 5/5.

So even in-library drag, the frozen numerical ruler is not a transported
ranking. Decision-bit agreement at 80% is the preregistered H1a floor, not
evidence that \(C\) is intervention-invariant.

## Combined with RS2A

\[
\boxed{
\begin{aligned}
\text{physics representation (}J^\top f\text{ accounting) transports}\\
\text{detectability calibration does not (RS2A)}\\
\text{consequence calibration does not (RS2B, out-of-library harm)}
\end{aligned}
}
\]

Or:

\[
\boxed{
\text{physics representation can transport across interaction mechanisms,
while epistemic decision statistics may require transport calibration}
}
\]

World-model self-improvement policy inputs \((D_0,C,V)\) are themselves
domain-specific. RS2 is not a robosuite contact bug.

## Status

\[
\boxed{
\begin{aligned}
RS1C\_GO&=\mathrm{true}\\
RS2\text{-C0\_PASS}&=\mathrm{true}\\
RS2\_GO&=\mathrm{false}\\
RS2A\_GO&=\mathrm{true}\\
RS2B\_GO&=\mathrm{true}
\end{aligned}
}
\]

No \(C_{\mathrm{tol}}\) search. No detectability reopen. No `RS2_GO` flip.
