# R1-RS1A.5 Preregistration — Value-of-Epistemic-Excitation

Date: 2026-08-15  
Status: **FROZEN** — formal completed; see `REPORT/REP/R1/R1_RS1A5_REPORT.md` (`RS1A.5_GO=true`; \(A^\star=1.5A_0\))

## Placement

\[
\boxed{
\begin{aligned}
RS1A.4 &\colon \text{misses = tolerable }\cup\text{ consequential probe states}\\
RS1A.5 &\colon \textbf{when is epistemic excitation worth its cost?}\\
RS1B &\colon \text{H32 stability on revise-worthy only (still locked)}
\end{aligned}
}
\]

## Permanently frozen policy populations (from RS1A.4)

\[
\boxed{
\begin{aligned}
\text{tolerate} &= \{C < C_{\mathrm{tol}}\}\\
\text{probe} &= \{C \ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=0\}\\
\text{revise-worthy }(\mathcal P_{\mathrm{rev}}) &= \{C \ge C_{\mathrm{tol}}\}\cap\{\mathrm{detect}=1\}
\end{aligned}
}
\]

RS1B, when eventually opened, uses **only** \(\mathcal P_{\mathrm{rev}}\).

Frozen detector stack unchanged: \(E_{\mathrm{known}}=D_0\), support offline, typed channels.

## Scientific question

\[
\boxed{
\text{Given a probe-state mismatch, should we spend action cost to excite it?}
}
\]

Not “larger \(A\) always helps” (already shown in RS1A.3), but:

\[
\boxed{
\text{Is the information gained from excitation worth its intervention cost?}
}
\]

## Scope

**Only** the probe population. Exclude tolerate and already-detected cases from VoI aggregation.

## Grid

| Factor | Levels |
|--------|--------|
| \(\alpha\) | \(\{0,-0.18,-0.24\}\) (\(0\) = calibration / C0) |
| \(A/A_0\) | \(\{0.5,1.0,1.5,2.0\}\), \(A_0=0.12\), \(f=0.20\) Hz |
| Seeds | \(\{9801,9811,9821,9831,9841\}\) |

\[
5\times3\times4=\boxed{60\ \mathrm{trajectories}}
\]

Smoke: seed `9001` × \(\{0,-0.24\}\) × \(\{0.5A_0,2A_0\}\).

## Frozen quantities

Same \(D_0\) (probe trajectory), \(C=L_{\mathrm{no\text{-}rev}}-L_{\mathrm{adeq}}\) (queries depend only on seed+\(\alpha\)), cell thresholds at C0 FPR\(\le1\%\).

\[
C_{\mathrm{tol}}
=
\mathrm{median}\{C(\alpha{=}0)\}
+
0.5\cdot\mathrm{median}\{C(\alpha{=}{-}0.24,A{=}2A_0)\}
\]

## Action cost and VoI (frozen)

Action intensity:

\[
c(A)=\bigl(A/A_0\bigr)^2
\]

Price of intervention (a priori):

\[
\lambda = 0.0015
\]

chosen so \(\lambda\,c(2A_0)=0.006\) is on the same order as \(C(\alpha{=}{-}0.24)\) from RS1A.4.

For each probe instance at baseline \(A_b\in\{0.5,1.0\}A_0\), and candidate \(A_a>A_b\):

\[
\begin{aligned}
\mathrm{flip}(A_a)
&=
\mathbf{1}[\mathrm{detect}(A_a)=1]\\
V(A_a\mid A_b)
&=
\mathrm{flip}(A_a)\cdot C
-
\lambda\, c(A_a)
\end{aligned}
\]

Secondary (report): incremental cost

\[
V_{\Delta}(A_a\mid A_b)
=
\mathrm{flip}\cdot C
-
\lambda\bigl(c(A_a)-c(A_b)\bigr).
\]

## Primary endpoints

1. Seed-mean \(V\) for each \((A_b,A_a)\) on the probe set  
2. Best action \(A^\star(A_b)=\arg\max_{A_a} \overline V(A_a\mid A_b)\)  
3. \(P(\mathrm{flip}\mid A_a)\) (information gain proxy)

## Hard gates

\[
\boxed{
\exists\,A_a:
\overline V(A_a\mid 0.5A_0) > 0
}
\]

\[
\boxed{
A^\star(0.5A_0)
\text{ is well-defined and }
\overline V(A^\star)>0
}
\]

\[
\boxed{
\text{report whether }A^\star=\max A
\text{ or an interior / cheaper action wins after cost}
}
\]

(third is descriptive; GO uses the first two)

Interpretation:

| Result | Meaning |
|--------|---------|
| \(V>0\) for some finite boost | epistemic excitation can be worth paying |
| \(A^\star<\max A\) | cost matters; not “always blast” |
| all \(V\le0\) | either \(\lambda\) too harsh or probe→revise conversion too unreliable under cost |

## Non-goals

- Opening RS1B
- Retuning \(D_0\) / support
- VoI on tolerate cases
- Maximizing recall for \(|\alpha|\le0.09\)
