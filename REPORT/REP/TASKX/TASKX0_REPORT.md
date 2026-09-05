# TASK-X0 Family Report — Oracle Progress Instrument (3 tasks)

Date: 2026-08-27
Status: **RAN**; **family FAIL** (`task_x0_passed=false` on all three);
**TASK-X1 stays LOCKED**; **TASK-XL FROZEN after instrument_ready** (not run)
Prereg: `REPORT/REG/TASKX/TASKX0_PREREG.md`
Retry (design only): `REPORT/REG/TASKX/TASKXL_PREREG.md`
Per-task: `TASKX0_CUP_REPORT.md`, `TASKX0_CABINET_REPORT.md`,
`TASKX0_STAMP_REPORT.md`
Does not: diffusion / TASK-X1; RGB; R10; RoboTwin-X0 `physics_predict`;
\(S_{\mathrm{task}}\) claim; retune frozen numeric gates; relabel run JSON
`task_x0_passed`; start TASK-XL0 / interrupt RoboTwin-X0R

## Frozen interpretation (do not collapse)

\[
\boxed{
\textbf{TASK-X0 is an instrument / representation test that did not PASS.}
\text{ It is not a warrant to write that task progress has no planning value.}
}
\]

Two failure classes **must stay separate**:

1. **Data / execution chain** — cabinet `coverage_hole` is the prototype
   (almost no grasp/transport; HDF5 missing object pose; CuRobo /
   `play_once` unusable). This only supports:
   **TASK-X0 instrument insufficient.**
   It does **not** support: **\(p_t\) has no decision value.**

2. **Genuine negative (this probe)** — cup and stamp G0 / G1 / G3 PASS
   (causal \(G\); phases not fully degenerate) but frozen kNN/NLL: adding
   \(p_t\) does not significantly reduce the proxy of \(H(A\mid s,g)\).
   Stamp G2 CI is entirely negative. This supports:
   **\(p_t\) did not further compress action uncertainty**
   (this version of explicit progress was not validated).

## One-line

Three official-task instruments **finished**. None PASSed. Cup/stamp:
`p_no_nll` (G2 on a working causal \(G\)). Cabinet: `coverage_hole`
(instrument, not a \(p_t\)-value verdict). Do **not** open TASK-X1.
Do **not** rewrite “progress unproven” as “need a stronger generator.”

## Per-task (frozen gates, not retuned)

| task | pattern | G0 | G1 | G2 \(\Delta\mathrm{NLL}\) [CI] | G3 | `check_success` |
|---|---|---|---|---|---|---|
| `place_empty_cup` | `p_no_nll` | pass | pass | \(-0.47\) \([-1.00,0.27]\) **fail** | pass | **0/50** |
| `put_object_cabinet` | `coverage_hole` | **fail** | fail | \(-0.52\) (n_holdout=66) **fail** | pass | **1** episode kept |
| `stamp_seal` | `p_no_nll` | pass | pass | \(-0.48\) \([-0.68,-0.29]\) **fail** | pass | hdf5 50 (no play_once) |

G3 `future_leak=false` on all three. Oracle \(G\) is causal.

## Two failure layers (do not collapse)

**Layer 1 — Data / execution chain (instrument insufficient).** CuRobo /
pytorch3d missing; `play_once` not a success source. Official `demo_clean`
hdf5 is **qpos/endpose**, typically **no object/cabinet pose**. Cup replay
does not re-grasp (`check_success=0/50`). Cabinet hits hollow `077_phone`
/ missing `model_data`. Phase labels on cup/stamp are predicates on
robot/EE/init objects, **not** verified manipulation success. Cabinet
never populated grasp/transport/release. **Prototype: cabinet
`coverage_hole`.** This class licenses only that the TASK-X0 instrument
was insufficient. It does **not** license “\(p_t\) has no decision value.”

**Layer 2 — Genuine negative of this G2 probe.** Where coverage exists
(cup, stamp), G0 / G1 / G3 PASS. Adding oracle \(p\) **does not** reduce
held-out action NLL; stamp CI is strictly negative. That is a fail of
*this* frozen density probe on *this* action chunk: explicit \(p_t\) did
not extra-compress the proxy of \(H(A\mid s,g)\). Not a TASK-X1
\(S_{\mathrm{task}}\) result. Not a reason to swap in a stronger
generator.

G1 aliasing on cup/stamp is weak in action gap (same-\(p\) NN already
small). Do not over-read “\(s\)-aliasing proves planning value.”

## PLAN-X parallel (what stays closed)

PLAN-X already showed that **state + goal can learn a search center**
(\(\mu_\phi\)). TASK-X0 has **not** shown that an explicit progress state
further improves the action distribution. Therefore remaining **closed**:
diffusion, TASK-X1. Do **not** convert “progress unproven” into “need a
stronger generator.”

## Unlock / priority (freeze)

Family PASS required all three `task_x0_passed`. **None** did.
TASK-X1 / TASK-X2 / PLAN-X2 / R10 unchanged (locked). Do not retune
frozen numeric gates. Do not relabel run JSON `task_x0_passed`.

**Priority (information gain, freeze):**

```text
RTWX-X0S structure audit  >  φ-ID  >  capacity  >  TASK-XL  >  visual  >  diffusion
```

**X0R = RAN / `excitation_failure`.** \(\Delta s\neq 0\); G1 audit PASS.
Identity still beats Euler; \(\hat\phi\) worse than fixed \(\phi\). Next
is **structure closure** (X0S), not XL0. Capacity still **WITHHELD**.

If TASK-X is retried later: first fix **coverage and object-state
logging**, not swap models. The retry representation is **learned**
\(z_g,z^p\) vs a **no-progress** baseline (`TASKXL_PREREG.md`), **not**
a rewritten hand phase table. Cabinet `coverage_hole` remains an
instrument hole; cup/stamp G2 on frozen kNN does **not** prove learned
\(z^p\) worthless. TASK-X1 diffusion stays **LOCKED**. This document
does not implement an X0R or XL0 runner.

## Ledger

```text
RoboTwin-X0
= RAN / capacity claim WITHHELD
原因：
- A/C 接近 identity predictor
- residual 反而恶化
- φ 未辨识
- latent rollout 不稳定

TASK-X0
= FAIL
cup    : p_no_nll
stamp  : p_no_nll   ← 当前探针下是真负结果
cabinet: coverage_hole ← 仪器不足
TASK-X1 = LOCKED

RoboTwin-X0R
= FAIL / excitation_failure
G1 force-auditable = PASS
G0 physics > identity = FAIL
G2 φ-identification = FAIL
G3 latent rollout sanity = FAIL

RTWX-X0S = RAN / FAIL  timing_mismatch
RTWX-X0F = RAN / FAIL  force_channel_unresolved
TASK-XL  = DESIGN FROZEN ONLY; LOCKED
R10      = LOCKED
```
