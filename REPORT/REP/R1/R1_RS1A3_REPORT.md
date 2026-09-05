# R1-RS1A.3 Report — Excitation-Limited Inadequacy Identifiability

Date: 2026-08-15  
Prereg: `REPORT/REG/R1/R1_RS1A3_PREREG.md`  
Artifacts: `runs/r1_rs1a3/formal/` (`excitation_scores.csv`, `cells.csv`, `summary.json`)

## Decision

\[
\boxed{RS1A.3\_GO = \mathrm{true}}
\]

\[
\boxed{
\text{C1-L weak-signal miss is excitation-limited under frozen }D_0
}
\]

RS1B remains **locked**. Support typed channel stays frozen offline.  
Next scientific stage unlocked for drafting: **RS1A.4** (detectability ↔ consequence).

## Confirmatory gates

| Gate | Value | Pass |
|------|------:|:----:|
| \(\overline R(2A_0)>\overline R(0.5A_0)\) | **0.80 > 0.00** | ✓ |
| Spearman\((X_\phi,\mathrm{detect})_{\mathrm{C1\text{-}L}}>0\) | **+0.559** (seed-mean **+0.903**) | ✓ |

## Seed-mean C1-L recall vs amplitude (avg over \(f\))

| \(A/A_0\) | Seed-mean recall @ cell FPR≤1% |
|--------:|-------------------------------:|
| 0.5 | **0.00** |
| 1.0 | 0.13 |
| 1.5 | 0.60 |
| 2.0 | **0.80** |

Monotone in amplitude. C0 mean \(D_0\) stays ≈ `0.002` across the grid (noise floor); excitation does **not** inflate adequate-physics false alarms.

## Cell map (AUROC / recall / \(X_\phi\))

| A | f (Hz) | AUROC | Recall | mean \(X_\phi\) (C1-L) |
|--:|-------:|------:|-------:|-----------------------:|
| 0.5 | 0.2/0.4/0.8 | ~0.48–0.50 | 0.00 | ~2e-15 |
| 1.0 | 0.2 | 0.56 | 0.20 | 6.4e-7 |
| 1.0 | 0.4 | 0.44 | 0.00 | 6.1e-8 |
| 1.0 | 0.8 | 0.52 | 0.20 | 5.0e-9 |
| 1.5 | 0.2 | **1.00** | **1.00** | 9.4e-4 |
| 1.5 | 0.4 | 0.84 | 0.60 | 1.4e-4 |
| 1.5 | 0.8 | 0.60 | 0.20 | 1.3e-5 |
| 2.0 | 0.2 | **1.00** | **1.00** | 6.2e-3 |
| 2.0 | 0.4 | **1.00** | **1.00** | 2.1e-3 |
| 2.0 | 0.8 | 0.88 | 0.40 | 1.7e-4 |

Low frequency at fixed amplitude yields **higher** \(X_\phi=\mathrm{mean}(v^4)\) and higher detectability — consistent with needing velocity support for \(\phi(v)=|v|v\), not merely large torque chatter.

## Mechanism reading

```text
low exposure  →  |r_τ|≈|α|v²≈0  →  D0 overlaps C0 noise floor  →  miss
high exposure →  operator region visited  →  D0 separates cleanly →  detect
```

This falsifies “need a smarter scalar on the same uninformative trajectories” as the remaining C1-L bottleneck, and confirms RS1A.2’s diagnosis:

\[
\boxed{
p(D_0\mid C1\text{-}L)
\text{ overlaps }
p(D_0\mid C0)
\text{ when excitation is insufficient}
}
\]

Logistic sketch on episodes: \(\mathrm{logit}\,P(\mathrm{detect})\approx a X_\phi+b\) with \(a>0\) (see `aggregate.logistic`).

## Frozen architecture (unchanged)

\[
\begin{aligned}
E_{\mathrm{known}} &\leftarrow D_0=\|r_\perp\|\\
E_{\mathrm{unknown}} &\leftarrow \text{support violation (offline)}
\end{aligned}
\]

Do **not** reopen support fusion or alternative residual statistics for C1-L.

## Unlock status

| Stage | Status |
|-------|--------|
| RS1A.3 | **passed** (excitation-limited confirmed) |
| RS1A.4 | unlocked for prereg (detectability vs consequence / tolerance) |
| RS1B | **locked** |
| RS2 | locked |

## Next (RS1A.4)

Completed: see `REPORT/REP/R1/R1_RS1A4_REPORT.md`. The tolerate / probe / revise_worthy
partition is empirically visible; RS1B stays locked until opened on
revise_worthy only.
