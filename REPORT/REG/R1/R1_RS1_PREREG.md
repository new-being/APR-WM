# R1-RS1 Preregistration — Frozen R0.6 → robosuite/MuJoCo Door

Date: 2026-08-15  
Status: **FROZEN** — formal 200-trajectory run completed; see `REPORT/REP/R1/R1_RS1_REPORT.md` (`RS1_GO=false`)

## Scientific question

\[
\boxed{
\text{Does frozen R0.6 retain model-inadequacy detection,
structure selection, physical rejection and useful revision
under robosuite/MuJoCo dynamics?}
}\]

Robot contact is excluded. This is a scientific transfer experiment, not an
infrastructure closure test.

## Unlock prerequisites (satisfied)

- R1-MS0: `infrastructure_block` (retained; not scientific no-go)
- R1-MJ0: passed
- R1-RS0: passed

## Frozen packages / scene

| Item | Value |
|------|-------|
| robosuite | `1.5.2` |
| MuJoCo | `3.11.0` |
| Environment | `Door` |
| Robot | Panda present, kinematically frozen |
| Drive | direct generalized torque on `Door_hinge` |
| Base profile | RS0 P0 (`frictionloss=0.10`, `damping=0.10`) |
| simulation dt | `0.002 s` |
| episode length | `10 s` |
| `include_constraint` | `True` |
| truth `qfrc_passive` | `raw_truth` only |
| R0.6 mechanism | frozen (no MuJoCo-specific operators added) |

## Regimes

| Regime | Truth | Expectation |
|--------|-------|-------------|
| C0 | adequate P0 | no revision |
| C1-L | \(r_\tau=-0.12\|v\|v\) | recovery |
| C1-H | \(r_\tau=-0.24\|v\|v\) | OOD coefficient recovery |
| CNEG | \(r_\tau=+0.12\|v\|v\) | evidence OK, physical reject |
| C2-latch | `use_latch=True` | unknown / reject wrong model |

Hidden force \(\tau_{\mathrm{hidden}}=\alpha|v|v\) is logged only under
`raw_truth`, never under `learner_visible`.

## Probe bank (fixed; no active redesign)

| ID | Type |
|----|------|
| P1 | low-frequency sine #1 |
| P2 | low-frequency sine #2 |
| P3 | high-frequency sine #1 |
| P4 | high-frequency sine #2 |
| P5 | upward chirp |
| P6 | downward chirp |
| P7 | slow piecewise-random |
| P8 | fast piecewise-random |

Amplitude follows the RS0-safe excitation scale (`0.12`). Random probes are
deterministic given the episode seed.

## Formal matrix

\[
\{9101,9111,9121,9131,9141\}
\times
5\ \mathrm{regimes}
\times
8\ \mathrm{probes}
=
200\ \mathrm{trajectories}
\]

Plumbing smoke (non-scientific): seed `8901` × 5 regimes × 1 probe.
Smoke may not change thresholds.

## Primary gates

| Gate | Criterion |
|------|-----------|
| C0 false revision | \(\le 1\%\) |
| C1 exact recovery (pooled, also report L/H) | \(\ge 85\%\) |
| C1 coefficient relative error \(\le 10\%\) among correct accepted | \(\ge 85\%\) |
| CNEG accepted active revision | **0** |
| C2 \(P(\mathrm{unknown}\mid\mathrm{triggered,evaluable})\) | \(\ge 90\%\) |
| C2 \(P(\mathrm{wrong}\mid\mathrm{triggered,evaluable})\) | \(\le 5\%\) |
| accepted H32 stability | \(\ge 99\%\) |
| C1 seed-level mean paired H32 gain | \(> 0\) (report \(t_4\) 95% CI) |

H2/H4/H8 are acceptance-visible. **H32 is blind** and cannot enter any
acceptance decision.

Utility statistics use **seed-level** aggregation (\(n=5\)), not trajectory-level
pseudo-N.

## Decision pipeline (ordered)

```text
observation → trigger → proposal → selection
  → H2/H4/H8 validation → physical admissibility → utility
  → ACCEPT/REJECT → freeze → H32 blind rollout
```

## GO rule

\[
RS1\_GO =
C0_{\mathrm{specific}}
\land C1_{\mathrm{recover}}
\land CNEG_{\mathrm{safe}}
\land C2_{\mathrm{reject}}
\land H32_{\mathrm{stable}}
\land H32_{\mathrm{useful}}
\]

On GO: unlock R1-RS2. On fail: diagnose by funnel layer; do **not** retune R0.6.
