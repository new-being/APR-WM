# R1-RS1A.4 Preregistration — Detectability vs Consequence

Date: 2026-08-15  
Status: **FROZEN** — formal completed; see `R1_RS1A4_REPORT.md` (`RS1A.4_GO=false`, tolerate/probe/revise validated)

## Placement

\[
\boxed{RS1A.3 \rightarrow RS1A.4 \rightarrow RS1B}
\]

RS1A.3 established: C1-L misses are **excitation-limited** under frozen \(D_0\).

## Frozen detector stack (do not reopen)

\[
\boxed{
\begin{aligned}
E_{\mathrm{known}} &= D_0=\|r_\perp\|
\quad\text{(probe-trajectory)}\\
E_{\mathrm{unknown}} &= \text{support violation (offline)}\\
&\text{typed channels; no unified scalar fusion}
\end{aligned}
}
\]

No revision acceptance / passivity / H32 stability study. RS1B locked.

## Scientific question

\[
\boxed{
\text{When should a world model spend epistemic effort on a small mismatch?}
}
\]

Operationally:

\[
\boxed{
\text{When is insufficient excitation actually a problem?}
}
\]

## Joint endpoints

### Detectability

\[
R(\alpha,X_\phi)=P(D_0>\tau\mid\alpha,X_\phi)
\]

with \(\tau\) calibrated at C0 FPR \(\le1\%\) **within each excitation cell**
(same construction as RS1A.3; \(\alpha=0\) plays the C0 role).

### Consequence

From identical ICs, integrate horizon \(H=32\) macro-steps (same macro as R0.6/RS1):

\[
\begin{aligned}
L_{\mathrm{no\text{-}rev}} &= \mathrm{RMSE}(\hat x_{\mathrm{nominal}},x_{\mathrm{truth}})\\
L_{\mathrm{adeq}} &= \mathrm{RMSE}(\hat x_{\mathrm{oracle\text{-}drag}},x_{\mathrm{truth}})\\
C &= L_{\mathrm{no\text{-}rev}}-L_{\mathrm{adeq}}
\end{aligned}
\]

Oracle drag applies the same \(\alpha|v|v\) force as truth (identifiable upper bound).  
Report mean \(C\) over a frozen query set of 8 ICs per episode.

## Grid

| Factor | Levels |
|--------|--------|
| \(\alpha\) | \(\{0,-0.03,-0.06,-0.09,-0.12,-0.18,-0.24\}\) |
| \(A/A_0\) | \(\{0.5,1.0,1.5,2.0\}\), \(A_0=0.12\) |
| \(f\) | \(0.20\) Hz (highest exposure family from RS1A.3) |
| Seeds | \(\{9701,9711,9721,9731,9741\}\) |

\[
5\times7\times4 = \boxed{140\ \mathrm{trajectories}}
\]

Smoke: seed `8991` × \(\{\alpha=0,-0.12\}\) × \(A_0\).

## Harm / tolerance threshold (frozen a priori)

\[
C_{\mathrm{tol}}
=
\mathrm{median}\{\,C(\alpha{=}0)\,\}
+
0.5\cdot
\mathrm{median}\{\,C(\alpha{=}{-}0.24,\,A{=}2A_0)\,\}
\]

i.e. half of the strong-mismatch, high-excitation consequence above the \(\alpha=0\) floor.

## Policy regions (descriptive + confirmatory)

\[
\boxed{
\begin{array}{ll}
C<C_{\mathrm{tol}} &\Rightarrow \textbf{tolerate}\\[2mm]
C\ge C_{\mathrm{tol}},\ D_0\le\tau &\Rightarrow \textbf{probe (need excitation)}\\[2mm]
C\ge C_{\mathrm{tol}},\ D_0>\tau &\Rightarrow \textbf{revise-worthy}
\end{array}
}
\]

## Hard gates

\[
\boxed{
P(C<C_{\mathrm{tol}}\mid \mathrm{miss})
\ge
0.70
}
\]

(most misses are tolerated mismatches)

\[
\boxed{
P(\mathrm{detect}\mid C\ge C_{\mathrm{tol}})
>
P(\mathrm{detect}\mid C<C_{\mathrm{tol}})
}
\]

(consequence concentrates detectability)

\[
\boxed{
\text{mean }C\text{ among misses}
<
\text{mean }C\text{ among detects}
}
\]

GO = all three. Soft report: fraction of consequential episodes that remain misses (should be small); plot \(R\) and \(C\) vs \((\alpha,X_\phi)\).

## Unlock

On GO: RS1B may be drafted only on the **revise-worthy** population
\((C\ge C_{\mathrm{tol}}\land\mathrm{detect})\), not on all triggers.  
On fail: either \(C_{\mathrm{tol}}\) calibration or H32 consequence proxy needs redesign — still no RS1B on raw detections.
