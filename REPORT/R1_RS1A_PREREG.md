# R1-RS1A Preregistration — Multi-Evidence Model-Inadequacy Detection

Date: 2026-08-15  
Status: **FROZEN** — formal run completed; see `R1_RS1A_REPORT.md` (`RS1A_GO=false`, C2 support-coupling confirmed)

## Placement in the chain

RS1 showed differential transferability of APR-WM mechanisms under MuJoCo/Door:

| Mechanism | Transfer |
|-----------|----------|
| physics representation / force closure | strong |
| operator proposal | strong |
| passivity / physical admissibility | strong |
| weak inadequacy detection | weak |
| outside-support → inadequacy coupling | missing |
| short → long utility calibration | weak |

RS1 is therefore **not** “R0.6 failed to generalize.” It splits into two separate scientific questions:

\[
\boxed{R1\text{-RS1A Trigger}\ \rightarrow\ R1\text{-RS1B Stability}\ \rightarrow\ R1\text{-RS1C integrated confirmation}}
\]

**R1-RS2 stays locked** until RS1C. RS1A and RS1B must not be merged into one confirmatory experiment.

## Scientific question

\[
\boxed{
\text{Does structural detection require normalized,
persistent, multi-source evidence?}
}

\]

Core hypothesis:

> A transferable model-inadequacy detector should accumulate normalized
> persistent residual evidence together with model-support violations,
> rather than triggering from residual magnitude alone.

This is an epistemic / V3-line question. It is **not** a revision-recovery
optimization study.

## Explicitly frozen (do not touch)

The following remain frozen and are **not executed** for decisions in RS1A:

- R0.6 proposal / selection / ranking
- H2/H4/H8 validation
- passivity / physical admissibility
- utility gate
- acceptance / assimilation
- H32 rollout
- operator library
- MuJoCo-specific operators (still forbidden)

RS1A only compares **trigger evidence formulations**.

## Detectors (only comparison axis)

| ID | Definition | Role |
|----|------------|------|
| \(D_0\) | \(\|r_\perp\|\) (RMS over discovery window) | RS1 / R0.6 baseline |
| \(D_1\) | \(\|r_\perp\|/(\sigma_{\mathrm{pred}}+\epsilon)\) | uncertainty-normalized |
| \(D_2\) | persistent accumulation of per-step \(D_1\) evidence | temporal persistence |
| \(D_3\) | \(D_2\) + support/validity evidence | multi-source |

Concrete frozen formulas:

\[
\begin{aligned}
r_\perp &= r - \mathrm{Proj}_{\mathrm{col}(J_\theta)}(r)\\
D_0 &= \mathrm{RMS}(r_\perp)\\
z_i &= \frac{|r_{\perp,i}|}{\sigma_{\mathrm{pred},i}+\epsilon},\quad
\epsilon=10^{-8}\\
D_1 &= \mathrm{RMS}(z)\\
\ell_i &= \tfrac12(z_i^2-1)\\
D_2 &= \max_{1\le t\le N}\sum_{i=1}^{t}\ell_i\\
u &= \frac{n_{\mathrm{unsupported}}}{n_{\mathrm{audit}}}
\quad\text{(frozen hinge-range support audit; see below)}\\
D_3 &= D_2 + \gamma\cdot\big(-\log(1-u+\epsilon)\big),\quad
\gamma=1
\end{aligned}
\]

Residual scores \(D_0\)–\(D_2\) use the same passive/discovery windows as frozen
R0.6. Support rate \(u\) uses a **frozen support audit** independent of residual
magnitude:

- \(n_{\mathrm{audit}}=32\) states with \(q\) on a uniform grid over the hinge
  interior \([q_{\lo}+\delta,q_{\hi}-\delta]\), \(\dot q=0\), \(\tau_{\mathrm{cmd}}=0\)
- each state is tagged `{modeled, boundary, unsupported}` by the same validity
  rule as RS1
- this is model-support evidence, not a second residual threshold

\(\sigma_{\mathrm{pred}}\) comes from the passive-context base posterior
predictive variance. \(\gamma=1\) is frozen a priori; it is **not** tuned on
RS1A formal data.

**Forbidden:** choosing thresholds to maximize C1 recovery or H32 utility.

## Regimes and labels

| Regime | Label for primary ROC/PR | Role |
|--------|--------------------------|------|
| C0 | negative | specificity |
| C1-L | known-structural positive | weak in-library signal |
| C1-H | known-structural positive | strong in-library signal |
| C2-latch | outside-library positive | support-coupled structural |
| CNEG | diagnostic only | specificity/safety; **not** a primary endpoint |

Primary evaluation tasks:

1. **Known structural:** C0 vs (C1-L ∪ C1-H)
2. **Outside library:** C0 vs C2-latch
3. **Pooled structural:** C0 vs (C1-L ∪ C1-H ∪ C2)

## Scene / data boundary (reuse RS1 infrastructure)

Frozen from RS1 / RS0:

- robosuite `1.5.2`, MuJoCo `3.11.0`, Door Mode-A, Panda frozen
- P0: `frictionloss=0.10`, `damping=0.10`
- `dt=0.002`, episode `10 s`, probe bank P1–P8 (RS0-safe amplitudes)
- learner sees only \((q,\dot q,\ddot q,\tau_{\mathrm{command}},\textrm{nominal physics})\)
- hidden force only in `raw_truth`

## Formal matrix (new held-out seeds)

RS1 formal seeds `{9101…9141}` are **contaminated for RS1A confirmatory
claims**. Use reserved:

\[
\boxed{9201,\ 9211,\ 9221,\ 9231,\ 9241}
\]

\[
5\ \mathrm{seeds}
\times
\{C0,C1\text{-}L,C1\text{-}H,C2\text{-}latch\}
\times
8\ \mathrm{probes}
=
\boxed{160\ \mathrm{primary\ trajectories}}
\]

Plus diagnostic-only:

\[
5\times\{CNEG\}\times 8 = 40
\quad\text{(recorded; excluded from primary AUROC/AUPRC)}
\]

Plumbing smoke (non-scientific):

```text
seed = 8951 × {C0,C1-L,C1-H,C2-latch,CNEG} × P1
```

Smoke may not change formulas, \(\gamma\), or operating-point rules.

## Primary endpoints (not recovery)

For each detector \(D_k\), report:

- AUROC / AUPRC on the three primary tasks
- Full ROC: trigger recall vs C0 false-trigger rate
- Operating point at **C0 FPR ≤ 1%** (closest achievable on the discrete curve):
  - C1-L recall
  - C1-H recall
  - C2 recall
  - pooled structural recall

Hypothesis contrast (confirmatory):

\[
\boxed{
\begin{aligned}
&\mathrm{AUROC}(D_3)>\mathrm{AUROC}(D_0)
\quad\text{on outside-library and pooled}\\
&\text{at C0 FPR}\le1\%:\ 
\mathrm{Recall}_{C1\text{-}L}(D_3)>\mathrm{Recall}_{C1\text{-}L}(D_0)
\ \land\
\mathrm{Recall}_{C2}(D_3)>\mathrm{Recall}_{C2}(D_0)
\end{aligned}
}
\]

Secondary (non-GO): \(D_1\) vs \(D_0\), \(D_2\) vs \(D_1\) ablations — to localize
whether normalization, persistence, or support coupling carries the gain.

Seed-level aggregation (\(n=5\)) for AUROC differences when reporting
uncertainty; do not treat 160 trajectories as independent CI units for the
main claim.

## What RS1A does **not** claim

- That higher trigger rate alone is success
- That better detection implies better revision recovery
- That H32 stability is solved
- Permission to unlock RS2

## After RS1A

| Outcome | Next |
|---------|------|
| \(D_3\) dominates on C2 / weak C1-L without C0 collapse | freeze winning evidence class; open **R1-RS1B** |
| Only \(D_1\) helps, not support | inadequacy ≠ support-coupling; redesign C2 coupling |
| Nothing beats \(D_0\) on ROC | reformulate structural residual itself (deeper than threshold) |

RS1B remains a **separate** dynamics-invariant / long-horizon study and must
not retune RS1A detectors post hoc.
