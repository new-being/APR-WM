# TASK-X0 Preregistration — Task-Progress Instrument

Date: 2026-08-27
Status: **FROZEN**; **formal RAN 2026-08-27**; **FAIL** (all three tasks);
TASK-X1 remains LOCKED. Honest class: cup/stamp = this G2 probe did not
compress \(H(A\mid s,g)\); cabinet = instrument insufficient
(`coverage_hole`), **not** “\(p_t\) has no decision value.”
Retry family: `REPORT/REG/TASKX/TASKXL_PREREG.md` (learned \(z_g,z^p\);
**FROZEN after `instrument_ready`**; not a phase rewrite; not run).
Report: `REPORT/REP/TASKX/TASKX0_REPORT.md`
Depends on: `REPORT/REG/TASKX/TASKX_PREREG.md`
Does not: train \(\epsilon_\phi\); B0–B3 planning; RGB; RoboTwin-X0
physics predictor; unlock TASK-X1; unlock R10; start TASK-XL0

## One question

\[
\boxed{
\text{On official RoboTwin demonstrations, is oracle task progress
a closed, non-leaky, non-redundant instrument for later
progress-conditioned action attractors?}
}
\]

X0 **must not** claim that progress improves task success or search.

\[
\boxed{
\textbf{Family FAIL = instrument / representation test did not PASS.}
\text{ Not a warrant that task progress has no planning value.}
}
\]

Keep two classes: (1) data/execution chain (`coverage_hole`) \(\Rightarrow\)
instrument insufficient, **not** \(p_t\) has no decision value;
(2) genuine negative of this kNN/NLL probe (`p_no_nll` on cup/stamp)
\(\Rightarrow\) this \(p_t\) did not further compress
\(H(A\mid s,g)\).

## Data (frozen intent)

Successful demonstrations for:

1. `place_empty_cup`
2. `put_object_cabinet`
3. `stamp_seal`

Use RoboTwin’s collector / saved qpos–endpose. Scale: declare
\(N_{\mathrm{demo}}\) in the runner header **before** first parse
(instrument, not 20–50-task sweep).

Oracle \(p_t=G_{\mathrm{frozen}}(s_{\le t},g)\) only. Phase sets as in
the family prereg. Anti-leakage: no future labels.

## Gates

Numeric floors below are **frozen now** (instrument), not retuned after
seeing histograms.

| id | requirement |
|---|---|
| **G0 coverage** | each phase \(k\) on each task: \(N_k\ge N_{\min}=32\) |
| **G1 non-redundancy** | exist pairs with \(\|s_i-s_j\|<\epsilon_s\) but \(p_i\neq p_j\), and median \(\|A_i-A_j\|\) on those pairs exceeds the same-\(p\) nearest-neighbor action gap (formula: \(\epsilon_s=0.05\times\mathrm{RMS}(s)\) on the train demos, computed once in the header log). Documents **task-state aliasing**: \(s_t\) alone is not a unique task state |
| **G2 entropy** | held-out NLL of a frozen-class probe (kNN or small density) satisfies \(\mathrm{NLL}(A\mid s,g,p)<\mathrm{NLL}(A\mid s,g)\); paired \(\Delta\mathrm{NLL}>0\) with 95% CI not crossing 0. Absolute entropy is **not** the claim |
| **G3 recoverability** | every \(p_t\) used in G0–G2 is a function of \((s_{\le t},g)\) only; a leak check (shuffle future frames into \(G\)) must **not** be required for coverage. Log `future_leak=false` |
| **G-label** | `runs/task_x0/`; no \(S_{\mathrm{task}}\) claim; `oracle_progress=true` |

`task_x0_passed` iff all gates.

## Patterns (instrument)

| pattern | meaning |
|---|---|
| `instrument_ready` | all gates |
| `no_aliasing` | G1 fail: \(s\) already disambiguates \(p\) |
| `p_no_nll` | G2 fail: \(p\) does not reduce action uncertainty |
| `coverage_hole` | G0 fail |
| `progress_leaks` | G3 fail; do not open X1 |

## Unlock

TASK-X0 **did not PASS.** Therefore:

\[
\text{TASK-X1 (diffusion attractors) stays LOCKED.}
\]

Do **not** treat family FAIL as a license to freeze \(H_a,T_{\mathrm{diff}}\)
and train. The **retry** is TASK-XL (learned \(z_g,z^p\) vs no-progress),
**after** RoboTwin-X0R and after object-state coverage if using
RoboTwin demos — **not** a new hand phase table and **not** TASK-X1.

Hypothetical (historical; did not fire):

\[
\text{TASK-X0 PASS}
\;\Rightarrow\;
\text{freeze TASK-X1 (}H_a,\;T_{\mathrm{diff}},\;\text{widths, budget, }L,\;\delta,\;\text{seeds)}
\text{ then train}
\]

## Ledger

```text
TASK-X0
├── cup      = FAIL  p_no_nll
├── cabinet  = FAIL  coverage_hole
└── stamp    = FAIL  p_no_nll

TASK-X1          = LOCKED (diffusion; X0 did not PASS)
TASK-XL          = FROZEN NEXT-after-X0R (learned z_g, z^p)
TASK-XL0         = LOCKED
TASK-X2          = LOCKED
PLAN-X2          = LOCKED
RoboTwin-X0R     = NEXT (highest IG)
R10              = LOCKED
```

Priority (freeze): RoboTwin-X0R dynamics instrument \(>\) TASK-X retry
\(>\) visual \(>\) diffusion. TASK-X retry **=** learned \(z_g,z^p\),
not rewrite phases. If using RoboTwin demos: first fix coverage and
object-state logging. Cup/stamp `p_no_nll` does **not** prove learned
\(z^p\) is worthless. Do not rewrite “progress unproven” as “need a
stronger generator.”
