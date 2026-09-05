# R1-RS1 Report — Frozen R0.6 Transfer on robosuite Door

Date: 2026-08-15  
Prereg: `REPORT/REG/R1/R1_RS1_PREREG.md` (frozen before formal run)  
Artifacts: `runs/r1_rs1/smoke/`, `runs/r1_rs1/formal/`

## Decision

\[
\boxed{RS1\_GO = \mathrm{false}}
\qquad
R1\text{-RS2 remains locked}
\]

Do **not** retune R0.6 or add MuJoCo-specific operators. Diagnose by funnel layer only.

## Plumbing smoke (seed 8901 × 5 × P1)

Non-scientific checks only:

| Check | Result |
|-------|--------|
| Crash / NaN / indexing | pass |
| C2 latch activation (`unsupported` samples) | pass |
| HDF5 schema | pass |
| `tau_hidden` only in `raw_truth` | pass |
| H32 blind (`decision` before H32) | pass |
| Residual ≈ hidden (leakage fix) | pass |

Scientific gates were **not** evaluated on smoke.

## Formal matrix

\[
\{9101,9111,9121,9131,9141\}
\times
\{C0,C1\text{-}L,C1\text{-}H,CNEG,C2\text{-}latch\}
\times
\{P1..P8\}
=
200\ \mathrm{trajectories}
\]

## Gate scorecard (prereg)

| Gate | Criterion | Value | Pass |
|------|-----------|------:|:----:|
| C0 false revision | ≤ 1% | **0.0%** | ✓ |
| C1 exact recovery (pooled) | ≥ 85% | **73.75%** (L **50%**, H **97.5%**) | ✗ |
| C1 coefficient ≤10% error | ≥ 85% of correct accepted | **84.7%** (50/59) | ✗ |
| CNEG accepted *active* (`abs_v_v`, \(\hat\alpha>0\)) | **0** | **0** (trigger 47.5%) | ✓ |
| C2 conditional unknown | ≥ 90% | **NaN** (`n_trig_eval=0`) | ✗ |
| C2 conditional forced-wrong | ≤ 5% | **NaN** | ✗ |
| accepted H32 stability | ≥ 99% | **53.5%** (38/71) | ✗ |
| C1 seed-level mean H32 gain | > 0 | **+0.00160** | ✓ |
| C1 seed-level \(t_4\) 95% CI | report only | \([-0.00055,\ +0.00375]\) | — |

## Regime funnels

Aggregated counts over 40 trajectories per regime:

```text
C0
  total 40 → evaluable 40 → triggered 0
  → no revision (specificity OK)

C1-L
  total 40 → evaluable 40 → triggered 21
  → truth_in_top3 40 → correctly_selected 21
  → physically_admissible 21 → utility_accepted 20 → h32_useful 6

C1-H
  total 40 → evaluable 40 → triggered 40
  → truth_in_top3 40 → correctly_selected 40
  → physically_admissible 40 → utility_accepted 39 → h32_useful 19

CNEG
  total 40 → evaluable 40 → triggered 19
  → truth_in_top3 40 → correctly_selected 0
  → physically_admissible 17 → utility_accepted 12 → h32_useful 3
  note: accepted cases use constrained ˆα=0 (not active positive drag)

C2-latch
  total 40 → evaluable 40 → triggered 0
  → mean p_unsupported ≈ 0.074 (latch present, but no inadequacy trigger)
  → conditional rejection undefined (vacuous)
```

## Funnel diagnosis (no retune)

| Failure | Layer | Evidence |
|---------|-------|----------|
| C1-L recovery 50% | **inadequacy detection / trigger** | `truth_in_top3=100%` but trigger only 52.5%; once triggered, exact≈95% |
| Coefficient gate 84.7% | **estimation** (near miss) | C1-H coef OK ≈95%; C1-L weaker signal → larger \(\|\hat\alpha-\alpha\|/\|\alpha\|\) |
| CNEG hard gate | **physical admissibility** OK | trigger 47.5%, active \(\hat\alpha>0\) acceptance = 0; zeroed-coef accepts occur but do not violate hard gate |
| C2 | **trigger miss (non-vacuity)** | latch unsupported samples exist, but trigger coverage = 0% → cannot claim model-class rejection |
| H32 stability 53.5% | **long-horizon utility / stability** | short-horizon accept often; H32 rollout of accepted revision frequently unstable |
| H32 mean gain > 0 | utility direction OK | confirmatory CI lower bound **not** > 0 (power/noise) |

Primary bottlenecks for this MuJoCo transfer:

1. **Trigger coverage** on weak / structural regimes (C1-L, C2)
2. **H32 stability** of accepted revisions (R0.3/R0.4-class long-horizon failure reappearing)
3. **C2 vacuity** — unsupported contact/constraint effect does not enter the inadequacy trigger

## CNEG note (safety)

Hard gate definition (prereg): accepted *active* physical revision = 0.

Observed:

- trigger rate 47.5%
- when `abs_v_v` is selected, passivity projects \(\hat\alpha\to 0\)
- therefore `cneg_accepted_active=0` (pass)

This is evidence that **passivity blocks the unsafe coefficient**, not that CNEG is invisible. Zero-coefficient “accepts” remain a reporting nuance for future diagnostics; they did not fail the frozen hard gate.

## Unlock status

| Stage | Status |
|-------|--------|
| R1-MJ0 | passed |
| R1-RS0 | passed |
| R1-RS1 | **failed** (scientific) |
| R1-RS2 | **locked** |

## Next (protocol)

RS1 is closed as a **differential transfer** result, not a single GO/NO-GO on
R0.6. Follow-on work is split:

\[
\boxed{R1\text{-RS1A Trigger}\ \rightarrow\ R1\text{-RS1B Stability}\ \rightarrow\ R1\text{-RS1C}}
\]

- **RS1A** (`REPORT/REG/R1/R1_RS1A_PREREG.md`): multi-evidence inadequacy detection; freeze
  revision/acceptance/H32; primary endpoint is trigger ROC/PR, not recovery.
- **RS1B** (`REPORT/REG/R1/R1_RS1B_PREREG.md`): locked draft until RS1A finishes; dynamics
  invariants for H32 failures — not “extend validation horizon.”
- **Do not** retune R0.6, add MuJoCo operators, or unlock RS2 from RS1 alone.

