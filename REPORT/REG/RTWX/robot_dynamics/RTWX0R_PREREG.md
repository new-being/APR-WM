# RoboTwin-X0R Preregistration — Excitation & Dynamics Interface Repair

Date: 2026-08-27
Status: **FROZEN**; **formal RAN 2026-08-27**; **`rtwx_x0r_passed=false`**;
pattern **`excitation_failure`**; G1 PASS; G0/G2/G3 FAIL. Report:
`REPORT/REP/RTWX/RTWX0R_REPORT.md`. Not `instrument_ready`. Next:
**RTWX-X0S** structure audit. Do not open capacity, TASK-XL0, X1, R10.
Depends on: RoboTwin-X0 RAN + capacity claim WITHHELD
  (`REPORT/REP/RTWX/RTWX0_REPORT.md`)
Does not: rescue X0 `physics_capacity_shift`; RGB / RoboTwin-X1; pick/stamp
sweep; capacity \(R_P\); unlock R10; retune X0 gates after seeing X0 curves

## One question

\[
\boxed{\textbf{RoboTwin-X0R — Excitation \& Dynamics Interface Repair}}
\]

\[
\boxed{
\text{Construct a dynamics benchmark on an official RoboTwin task
with non-trivial state change and auditable physics accounting.}
}
\]

This is a **new instrument cell**, not a salvage of X0 numbers.
**Highest information gain on the family ledger.** TASK-X retry (if any)
comes **after** X0R and is learned \(z_g,z^p\)
(`REPORT/REG/TASKX/TASKXL_PREREG.md`), not a rewritten phase table;
if retried, first fix coverage and object-state
logging. Do not open TASK-X1 / diffusion / RGB / R10
from this cell.

X0 showed: prediction accuracy is not a physics–capacity comparison
when trajectories are nearly static; `physics_predict` was near an
identity copy on A/C; \(\phi\) was not identified; latent rollouts
diverged with width.

## Task (one only)

**`put_object_cabinet` only** (articulated cabinet; clearest structure).

Pause `place_empty_cup` / `stamp_seal`. Those cells currently say
“drive is too weak for non-trivial dynamics,” not “physics is strong.”
Re-open contact only after G0–G3 PASS on cabinet.

## Forbidden

RGB, cameras in \(s\), X1, three-task compute, treating X0 JSON as PASS,
capacity curve before all four gates, R10.

## Four gates (must all PASS before any capacity curve)

Numeric \(\epsilon_{\mathrm{exc}},\delta,\tau_1,\tau_{10}\) are **frozen
in the X0R runner config before the first X0R collect**, not after
seeing X0R curves. They must put identity outside the X0 A/C
\(E_1\sim 10^{-3}\) copy regime. Do not copy X0 formal \(E_1\) as a
gate after the fact.

### G0 excitation

State change is not near zero:

\[
\operatorname{median}
\frac{\|s_{t+H}-s_t\|}
{\operatorname{scale}(s)}
>\epsilon_{\mathrm{exc}}.
\]

Identity must be a **weak** baseline:

\[
E_{\mathrm{identity}}
>
E_{\mathrm{oracle\text{-}physics}}
+\delta.
\]

Pass means: **copying state is no longer enough.**

### G1 physics closure

Cabinet joint: a real map

\[
(q,\dot q,\tau)\to\ddot q
\]

or generalized-force residual accounting.

If RoboTwin/SAPIEN cannot expose enough of \(q,\dot q,\ddot q,\tau\):

\[
\boxed{\text{this task is not force-auditable}}
\]

Stop. Do **not** continue a physics capacity claim on this task.

### G2 scene parameter identification

Estimate, CAP-X3-style,

\[
\phi=(m,I,b,\mu,\ldots)
\]

not a hardcoded \(\phi\) inside the predictor.

Require: \(D_{\mathrm{cal}}\to\hat\phi\) **improves** held-out query
prediction versus the frozen-\(\phi\) X0-style predictor.

### G3 latent rollout sanity

The **largest** latent baseline must satisfy both

\[
E_1<\tau_1
\qquad\text{and}\qquad
E_{\mathrm{roll10}}<\tau_{10}.
\]

Forbid the X0 extreme: best one-step width is worst rollout.
If G3 fails, **no** capacity curve.

## Patterns

| pattern | meaning |
|---|---|
| `instrument_ready` | G0–G3 all PASS; capacity curve still a **later** cell |
| `not_force_auditable` | G1 fail; stop physics-capacity on this task |
| `excitation_failure` | G0 fail |
| `phi_unidentified` | G2 fail |
| `latent_rollout_unstable` | G3 fail |
| `capacity_claim` | **forbidden in X0R** |

`rtwx_x0r_passed` iff `instrument_ready`. That unlocks writing a
**later** cabinet capacity cell. It does not unlock X1 or R10.

## Claims ceiling

X0R may say: official cabinet trajectories have non-trivial
\(\Delta s\), force/joint accounting is closed or documented as
impossible, \(\hat\phi\) from calibration helps, and a latent
baseline is rollout-sane.

X0R may **not** say: physics reduces latent capacity on RoboTwin;
X0 `physics_capacity_shift` is validated; pick/stamp are solved.

## Ledger

```text
CAP-X3            = PASS
PLAN-X            = CLOSED at iso_sufficient

RoboTwin-X0-P0    = PASS
RoboTwin-X0-smoke = PASS
RoboTwin-X0       = RAN
                    metrics finite
                    capacity claim WITHHELD
                    reason:
                    weak excitation /
                    identity-physics confound /
                    phi not identified /
                    latent rollout instability

RoboTwin-X0R      = RAN
                    G1 PASS (force-auditable)
                    G0/G2/G3 FAIL
                    pattern = excitation_failure
                    instrument_ready = false
                    next = RTWX-X0S (structure; not φ-ID first)
RTWX-X0S          = RAN / FAIL timing_mismatch
RTWX-X0F          = RAN / FAIL force_channel_unresolved
TASK-XL           = DESIGN FROZEN ONLY; LOCKED
TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll
TASK-X1           = LOCKED
PLAN-X2           = LOCKED
RoboTwin-X1       = LOCKED
R10               = LOCKED
```
