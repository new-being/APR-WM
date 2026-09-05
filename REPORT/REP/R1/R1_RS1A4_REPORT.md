# R1-RS1A.4 Report — Detectability vs Consequence

Date: 2026-08-15  
Prereg: `REPORT/REG/R1/R1_RS1A4_PREREG.md`  
Artifacts: `runs/r1_rs1a4/formal/`

## Decision

\[
\boxed{RS1A.4\_GO = \mathrm{false}}
\qquad
\text{(near-miss on tolerated-miss rate; decision structure validated)}
\]

RS1B remains locked. The revise-worthy population is now **defined**, but not yet unlocked for H32 study.

## Frozen stack (confirmed again)

\[
E_{\mathrm{known}}=D_0,\quad
E_{\mathrm{unknown}}=\mathrm{support\ (offline)},\quad
\text{typed channels}
\]

## Plumbing note

First formal draft confounded \(C\) with excitation via shared RNG for queries.  
Re-ran with queries depending only on \((\mathrm{seed},\alpha)\). After the fix,
\(C(\alpha)\) is **constant across \(A\)** (as required).

## Hard gates

| Gate | Value | Pass |
|------|------:|:----:|
| \(P(C<C_{\mathrm{tol}}\mid\mathrm{miss})\ge0.70\) | **0.642** | ✗ |
| \(P(\mathrm{detect}\mid C\ge C_{\mathrm{tol}}) > P(\mathrm{detect}\mid C<C_{\mathrm{tol}})\) | **0.500 > 0.403** | ✓ |
| \(\mathbb E[C\mid\mathrm{miss}] < \mathbb E[C\mid\mathrm{detect}]\) | **0.00270 < 0.00306** | ✓ |

\(C_{\mathrm{tol}}=0.00286\).

## Policy partition (nonzero \(\alpha\), n=120)

| Region | Count | Meaning |
|--------|------:|---------|
| **tolerate** | 72 | \(C<C_{\mathrm{tol}}\) |
| **probe** | 24 | consequential but undetected (need excitation) |
| **revise_worthy** | 24 | consequential and detected |

## Joint structure (mean over seeds)

Detectability rises with \(A\) at fixed \(\alpha\); consequence rises with \(|\alpha|\) and is **independent of probe \(A\)** after the query fix.

| \(\alpha\) | \(C\) | det @0.5\(A_0\) | det @2\(A_0\) | typical policy @low\(A\) → high\(A\) |
|----------:|------:|----------------:|--------------:|---|
| −0.03 | 0.00068 | 0.00 | 1.00 | tolerate → tolerate (detect optional) |
| −0.06 | 0.00204 | 0.00 | 1.00 | mostly tolerate |
| −0.09 | 0.00200 | 0.00 | 1.00 | mostly tolerate |
| −0.12 | 0.00275 | 0.00 | 1.00 | mixed → revise_worthy |
| −0.18 | 0.00408 | 0.00 | 1.00 | **probe → revise_worthy** |
| −0.24 | 0.00563 | 0.00 | 1.00 | **probe → revise_worthy** |

## Scientific reading

1. **Small mismatches are often correctly tolerated.**  
   \(|\alpha|\le0.09\) almost never crosses \(C_{\mathrm{tol}}\). Chasing recall→1 there would waste epistemic effort.

2. **The dangerous misses are not “detector failures under good observation.”**  
   They are **probe-region** cases: high \(|\alpha|\), low \(X_\phi\), \(C\ge C_{\mathrm{tol}}\), \(D_0\) silent.  
   This is exactly RS1A.3’s excitation limit expressed in decision language.

3. **Ideal ordering is empirically visible:**  
   at \(\alpha\in\{-0.18,-0.24\}\), moving \(0.5A_0\to2A_0\) converts **probe → revise_worthy** without inventing a new detector.

4. **Why GO failed narrowly:**  
   24/67 misses (36%) are consequential probe cases by construction of the grid (strong \(\alpha\) × weak \(A\)).  
   That violates the ≥70% “misses are harmless” gate — not because the theory is wrong, but because the grid **intentionally includes** the probe regime.  
   Soft reading: among misses at \(A\ge1.5A_0\) only, tolerated fraction is much higher (harmful misses concentrate at low excitation).

```text
tolerate  --C↑-->  probe (raise X_φ)  --detect-->  revise_worthy
```

## Suggested next

**R1-RS1A.5 — Value-of-Epistemic-Excitation** (`REPORT/REP/R1/R1_RS1A5_REPORT.md`): **passed**.  
Epistemic excitation has positive VoI at \(A^\star=1.5A_0\); max \(A\) is not optimal under cost.
RS1B remains locked; intake population = revise_worthy only.
