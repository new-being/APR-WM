# R1-RS1A.1 Report — Compositional Inadequacy Evidence

Date: 2026-08-15  
Prereg: `R1_RS1A1_PREREG.md`  
Artifacts: `runs/r1_rs1a1/formal/` (`gamma_selection.json`, `summary.json`)

## Decision

\[
\boxed{RS1A.1\_GO = \mathrm{false}}
\qquad
R1\text{-RS1B remains locked}
\]

## Scientific verdict (stronger than GO)

\[
\boxed{
\text{C1-L is not a support-coupling problem}
}
\]

On regimes with \(u=0\) (C0 / C1-L / C1-H / CNEG):

\[
D_3' \equiv D_1 \equiv D_g
\]

Therefore support composition **cannot** improve weak in-library detection
relative to \(D_1\), and in this held-out draw \(D_1\) does **not** beat \(D_0\)
at matched C0 FPR \(\le 1\%\). Support remains necessary for C2, but is the
wrong lever for C1-L.

## Protocol executed

1. **Dev γ selection** seeds `{9301,9311,9321}` on \(\gamma\in\{0.25,0.5,1,2,4\}\)  
   Objective: \(\max\min(R_{\mathrm{C1\text{-}L}},R_{\mathrm{C2}})\) s.t. C0 FPR≤1%  
   → all γ tied on C1-L recall (=0.25); selected **γ = 0.25** (smallest-γ tie-break)
2. **Formal held-out** seeds `{9401…9441}` × 4 primary × 8 probes (+ CNEG)  
   Revision / acceptance / H32 frozen off

## Hard gates (formal)

| Gate | Criterion | Value | Pass |
|------|-----------|------:|:----:|
| C1-L recall | \(R(D_3')>R(D_0)\) @ FPR≤1% | 0.125 ≯ 0.725 | ✗ |
| C2 recall | \(R(D_3')>R(D_0)\) @ FPR≤1% | **0.100 > 0.025** | ✓ |
| C1 AUROC non-degradation | \(AUROC(D_3')\ge AUROC(D_0)-0.02\) | 0.922 ≱ 0.953 | ✗ |

## AUROC (formal)

| Task | D0 | D1 | D3 (ref) | D3' | Dg |
|------|---:|---:|---------:|----:|---:|
| known structural (C1) | **0.973** | 0.922 | 0.687 | 0.922 | 0.922 |
| outside library (C2) | 0.464 | 0.606 | **1.000** | 0.896 | 0.896 |
| pooled | 0.803 | 0.817 | 0.791 | **0.914** | 0.914 |

## Operating point @ C0 FPR = 0

| Det | C1-L | C1-H | C2 |
|-----|-----:|-----:|---:|
| D0 | **0.725** | 1.000 | 0.025 |
| D1 / D3' / Dg | 0.125 | 0.775 | 0.100 |
| D3 (persistence+support) | 0.000 | 0.050 | **1.000** |

\(D_g\) vs \(D_3'\): max |diff| \(\sim 10^{-9}\) (as predicted while non-C2 have \(u=0\)).

## Mechanism reading

```text
magnitude (D0)     : best C1-L trigger at matched specificity
normalization (D1) : does not unlock C1-L here
persistence (D2/D3): still harmful for C1 ranking
support (→C2)      : validated again (D3 AUROC_C2=1; D3' helps C2 vs D0)
composition D3'    : cannot fix C1-L when u_C1=0
```

Research-line update:

\[
\boxed{
\begin{aligned}
&\text{magnitude}
\rightarrow
\text{normalization}
\rightarrow
\text{persistence falsified}
\rightarrow
\text{support coupling validated}
\\
&\rightarrow
\text{compositional add-on cannot rescue weak in-library signal}
\\
&\rightarrow
\text{next: weak-signal detector itself (not more γ/composition)}
\end{aligned}
}
\]

## What not to do next

- Do **not** open RS1B (trigger input distribution still C1-L–biased / unresolved)
- Do **not** keep sweeping γ or inventing new support mixes for C1-L
- Do **not** treat pooled AUROC (where D3' looks best) as a GO substitute

## Suggested next prereg (not RS1B)

**R1-RS1A.2 — Weak Structural Signal Detection** (`R1_RS1A2_PREREG.md`):

Compare magnitude vs SNR / direction / temporal correlation / library-matched
alignment. Keep support as a separate unknown channel. Do not resume \(\gamma\) fusion.

