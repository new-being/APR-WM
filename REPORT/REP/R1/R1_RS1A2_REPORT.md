# R1-RS1A.2 Report — Weak Structural Signal Detection

Date: 2026-08-15  
Prereg: `REPORT/REG/R1/R1_RS1A2_PREREG.md`  
Artifacts: `runs/r1_rs1a2/formal/`

## Decision

\[
\boxed{RS1A.2\_GO = \mathrm{false}}
\qquad
R1\text{-RS1B remains locked}
\]

## Scientific verdict

\[
\boxed{
\begin{aligned}
&\text{C1-L signal is carried primarily by residual magnitude }(D_0)\\
&\text{SNR / direction / temporal correlation }\approx\text{ chance}\\
&D_{\mathrm{lib}}\text{ is informative but strictly weaker than }D_0\\
&\text{typed }(D_{\mathrm{known}}\lor D_{\mathrm{support}})\text{ solves C2 without hurting C0}
\end{aligned}
}
\]

The operator-aligned hypothesis —

> weak inadequacy is easier to detect in operator-aligned space than in residual-energy space

— is **falsified as a way to beat \(D_0\)** on C1-L at matched specificity.  
\(D_{\mathrm{lib}}\) remains a useful secondary score (AUROC 0.81) but loses the hard recall gate.

## Matrix

Seeds `{9501…9541}` × `{C0,C1-L,C1-H,C2}` × 8 probes = 160 (+ CNEG diagnostic).  
No revision / acceptance / H32; no \(\gamma\) fusion.

## Primary task: C0 vs C1-L

| Statistic | AUROC | AUPRC | Recall @ C0 FPR≤1% |
|-----------|------:|------:|-------------------:|
| \(D_0\) | **0.943** | **0.945** | **0.475** |
| \(D_{\mathrm{lib}}\) | 0.806 | 0.848 | 0.325 |
| \(D_{\mathrm{SNR}}\) | 0.588 | 0.575 | 0.000 |
| \(D_{\mathrm{dir}}\) | 0.566 | 0.604 | 0.050 |
| \(D_{\mathrm{corr}}\) | 0.493 | 0.481 | 0.000 |

Hard GO (challenger beats \(D_0\) on C1-L recall + AUROC non-deg): **no qualifier**.

## Secondary: C1-H / pooled AUROC

| Det | C1-H | C1 pooled |
|-----|-----:|----------:|
| \(D_0\) | **1.000** | **0.972** |
| \(D_{\mathrm{lib}}\) | 0.981 | 0.894 |
| \(D_{\mathrm{SNR}}\) | 0.591 | 0.589 |
| \(D_{\mathrm{dir}}\) | 0.537 | 0.551 |
| \(D_{\mathrm{corr}}\) | 0.516 | 0.504 |

## Typed channels (secondary)

| System | C0 FPR | C1-L | C1-H | C2 |
|--------|-------:|-----:|-----:|---:|
| \(D_0 \lor D_{\mathrm{support}}\) | 0 | 0.475 | 1.000 | **1.000** |
| \(D_{\mathrm{lib}} \lor D_{\mathrm{support}}\) | 0 | 0.325 | 0.700 | **1.000** |

This confirms the architectural claim from RS1A/A.1:

\[
\boxed{
\text{known-structure and support-violation should be separate epistemic routes}
}
\]

even though the known route is still best served by \(D_0\) today.

## Mechanism notes

1. **Coherence ≠ missing information.** \(D_{\mathrm{dir}}\) / \(D_{\mathrm{corr}}\) near chance shows that “consistent structure across time” is not what currently separates C1-L from C0 under this discovery window — or that C0 residuals already have similar short-window sign/lag structure.

2. **Library match is partially right, often wrong.** On C1-L, `lib_best_operator` mass:
   - `signed_v2`: 16/40  
   - `abs_v_v`: 4/40  
   - others: x3/x2/saturation/…  
   So \(\max_{k\in\mathcal L}\) frequently locks onto a correlated quadratic velocity feature, diluting the true `|v|v` signature without ground-truth privilege (as required).

3. **Remaining C1-L miss under \(D_0\) is not “wrong statistic family” in this panel.** At FPR=0, \(D_0\) still only recalls 47.5% of C1-L — the residual energy overlap with C0 is real. Next levers are likely **sampling / excitation / window design**, not more scalar transforms of the same \(r_\perp\).

## Research-line update

\[
\boxed{
\begin{aligned}
RS1 &\colon \text{weak structural + outside-library trigger fail}\\
RS1A &\colon \text{support solves C2; naive persistence hurts C1}\\
RS1A.1 &\colon \text{support composition cannot solve C1-L}\\
RS1A.2 &\colon \text{C1-L lives in magnitude space; coherence/match do not beat }D_0
\end{aligned}
}
\]

Highest-level detector lesson:

\[
\boxed{
\text{model inadequacy is multi-causal and needs typed evidence channels}
}
\]

with current best known-channel statistic still:

\[
D_{\mathrm{known}} \approx D_0
\]

## Unlock status

| Stage | Status |
|-------|--------|
| RS1A.2 | informative fail (panel complete) |
| RS1B | **locked** |
| RS2 | locked |

## Suggested next (still not RS1B unless explicitly re-scoped)

**R1-RS1A.3 — Excitation-Limited Inadequacy Identifiability**
(`REPORT/REG/R1/R1_RS1A3_PREREG.md`): freeze \(D_0\) + typed support; vary amplitude/frequency;
test whether C1-L detectability tracks operator exposure \(X_\phi=\mathrm{mean}(v^4)\).

Do **not** return to \(\gamma\) support fusion for C1-L.

