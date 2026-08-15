# R1-RS1A Report — Multi-Evidence Model-Inadequacy Detection

Date: 2026-08-15  
Prereg: `R1_RS1A_PREREG.md`  
Artifacts: `runs/r1_rs1a/smoke/`, `runs/r1_rs1a/formal/`

## Decision

\[
\boxed{RS1A\_GO = \mathrm{false}}
\qquad
R1\text{-RS1B remains locked}
\]

This is **not** a null result. The detector ablation cleanly separates mechanisms.

## Scientific question (answered partially)

\[
\text{Does structural detection require normalized, persistent, multi-source evidence?}
\]

| Mechanism | Finding |
|-----------|---------|
| magnitude alone \(D_0\) | strong on known in-library (C1); **near-chance on C2** |
| uncertainty normalization \(D_1\) | small pooled gain; helps C2 modestly |
| persistence \(D_2\) as cumulative \(\sum\tfrac12(z^2-1)\) | **hurts** known-structural discrimination |
| support coupling \(D_3=D_2+\gamma(-\log(1-u))\) | **recovers C2 perfectly**; does not restore C1-L |

## Matrix

- Seeds `{9201…9241}` × `{C0,C1-L,C1-H,C2-latch}` × 8 probes = **160** primary  
- CNEG × 40 diagnostic only  
- Revision / acceptance / passivity / H32 **not executed**

## Primary metrics

### AUROC / AUPRC

| Task | Det | AUROC | AUPRC |
|------|-----|------:|------:|
| known structural (C0 vs C1) | D0 | **0.947** | 0.974 |
| | D1 | 0.919 | 0.963 |
| | D2 | 0.680 | 0.832 |
| | D3 | 0.680 | 0.832 |
| outside library (C0 vs C2) | D0 | 0.596 | 0.583 |
| | D1 | 0.711 | 0.748 |
| | D2 | 0.682 | 0.698 |
| | D3 | **1.000** | **1.000** |
| pooled structural | D0 | **0.830** | 0.943 |
| | D1 | 0.850 | 0.950 |
| | D2 | 0.681 | 0.874 |
| | D3 | 0.787 | 0.929 |

### Operating point (C0 FPR ≤ 1%; achieved FPR = 0)

| Det | C1-L recall | C1-H recall | C2 recall | pooled recall |
|-----|------------:|------------:|----------:|--------------:|
| D0 | 0.25 | **1.00** | **0.00** | 0.417 |
| D1 | **0.30** | 0.95 | 0.10 | 0.450 |
| D2 | 0.075 | 0.35 | 0.10 | 0.175 |
| D3 | 0.075 | 0.35 | **1.00** | **0.475** |

### Prereg hypothesis checklist

| Claim | Result |
|-------|:------:|
| AUROC\((D_3)>\)AUROC\((D_0)\) on outside-library | ✓ |
| AUROC\((D_3)>\)AUROC\((D_0)\) on pooled | ✗ (0.787 < 0.830) |
| C1-L recall\((D_3)>\)recall\((D_0)\) @ FPR≤1% | ✗ |
| C2 recall\((D_3)>\)recall\((D_0)\) @ FPR≤1% | ✓ (1.00 > 0.00) |

Seed-mean ΔAUROC (D3−D0): outside **+0.409**; pooled **−0.050**.

## Ablation localization

```text
D1 − D0  (pooled AUROC)   = +0.020   → normalization helps slightly
D2 − D1  (pooled AUROC)   = −0.169   → this persistence form hurts
D3 − D2  (outside AUROC)  = +0.318   → support coupling carries C2
```

Support audit (frozen): C2 \(u=0.156\) constant; all other regimes \(u=0\).

## Interpretation (V3-line)

1. **RS1’s C2 vacuity was a detector failure, not absence of structure.**  
   Once support violations enter the evidence, C2 is perfectly separable from C0 without touching revision.

2. **Weak in-library signal (C1-L) is not fixed by the same multi-source recipe.**  
   At the specificity-matched operating point, \(D_0\) already gets C1-H; C1-L remains the hard case. Persistence-as-implemented degrades the C1 ROC.

3. **Therefore the right next detector question is compositional, not “raise threshold”:**  
   keep support coupling for outside-library; replace or gate the persistence term so it cannot destroy C1 ranking.

4. Recovery / H32 remain out of scope — as preregged.

## Unlock status

| Stage | Status |
|-------|--------|
| R1-RS1 | closed (differential transfer) |
| R1-RS1A | **informative fail** on full GO; C2 mechanism confirmed |
| R1-RS1B | **locked** (do not open on partial C2-only success) |
| R1-RS2 | locked |

## Recommended next (still not RS1B)

**R1-RS1A.2 — Weak Structural Signal Detection** (`R1_RS1A2_PREREG.md`):

Compare \(D_0\) vs SNR / direction / temporal correlation / library-matched
alignment. Support stays a **separate** unknown channel. No \(\gamma\) fusion.


