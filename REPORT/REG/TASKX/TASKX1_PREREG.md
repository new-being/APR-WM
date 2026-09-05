# TASK-X1 Preregistration — Oracle Progress Action Attractors

Date: 2026-08-27
Status: **FROZEN as design**; **LOCKED** (TASK-X0 did not PASS).
TASK-XL is **not** this cell: retry is learned \(z_g,z^p\)
(`TASKXL_PREREG.md`), still below X0R, **without** opening diffusion.
Depends on: `REPORT/REG/TASKX/TASKX_PREREG.md`, TASK-X0 PASS
Does not: learned \(\hat p\) / TASK-XL0 train; RGB; PLAN-X2 on `capx_arm3`;
RoboTwin-X0 `physics_predict`; R10; retune X0 gates after X1 curves

## One question

\[
\boxed{
\text{With oracle }p_t\text{, does }q(A\mid s,g,p)\text{ raise task
success and search efficiency vs goal-only, history, and shuffled }p\text{?}
}
\]

## Matched protocol (intent; numbers after X0)

Same \(H_a\), diffusion steps, backbone, parameter budget, demos,
optimizer, test seeds. Primary variable: explicit \(p_t\).

B0 \(q(A\mid s,g)\); B1 \(q(A\mid s,g,p)\); B2 \(q(A\mid s_{t-L:t},g)\);
B3 shuffled \(p\). Require B1 > B3.

Scoring: RoboTwin simulator / env rollout of candidate chunks
(not the X0 identity-physics net). \(B\in\{1,2,4,8,16\}\).

Metrics M1–M3 + \(AUC_S\), \(B_{80}\) as family prereg.

## Gates

G-task, G-search, G-history as family. \(\delta\), \(L\), \(H_a\)
declared after X0 PASS, before train.

`task_x1_passed` iff those three and B1 > B3.

## Patterns

A `progress_planning_advantage` / B `history_sufficient` /
C `state_sufficient` (STOP) / D `progress_hurts` as family.

Pattern C or D: do **not** open TASK-X2 or scale tasks.

## Claims ceiling

May claim oracle progress as compressed planning state **if** Pattern A
and gates PASS. May not claim visual \(\hat p\), physics capacity, or
that PLAN-X should have used diffusion on the arm host.

## Ledger

```text
TASK-X1 = LOCKED until TASK-X0 PASS  (did not fire; stays LOCKED)
TASK-XL = FROZEN after instrument_ready; does not unlock this cell
TASK-X2 = LOCKED until TASK-X1 PASS
R10     = LOCKED
```
