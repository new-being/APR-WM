# CAP-X2-P0 Report — Oracle Planning Harness Lock

Date: 2026-08-18  
Status: **`cap_x2_p0_passed=true`**; **planning component DISABLED for X2**  
Prereg: `REPORT/REG/CAPX/CAPX2_PREREG.md`  
Artifacts: `runs/cap_x2/p0/`  
Does not: inspect PureNN/Hybrid capacity curves; expand budget grid; unlock R10

## Question

Select the smallest frozen CEM budget with oracle success \(\ge 0.80\) on
**reachable** targets, or disable planning for CAP-X2.

## Result

\[
\boxed{\text{planning disabled for X2}}
\]

| budget (cand, iters) | \(S_{\mathrm{oracle}}\) | mean terminal dist | CEM wall / task |
|---|---:|---:|---:|
| (256, 5) | 0.375 | 0.235 | 6.55 s |
| (512, 5) | 0.542 | 0.125 | 13.15 s |
| (512, 8) | 0.667 | 0.123 | 21.05 s |
| **(1024, 8)** max | **0.708** | 0.117 | 41.83 s |

Max oracle success \(0.708 < 0.80\) on 24 reachable-target tasks
(\(\rho=0\) plant). Per prereg: **do not enlarge the budget grid**.

## Freeze payload

`runs/cap_x2/p0/FROZEN_PLANNER.json`:

```json
{
  "planning_enabled": false,
  "cem_candidates": null,
  "cem_iters": null,
  "S_oracle": 0.7083333333333334,
  "reason": "max budget S_oracle < 0.80; planning disabled for X2"
}
```

## Consequence for CAP-X2

Matched rule becomes:

\[
M = M_1 \land M_2
\]

(one-step + rollout only). Reports must state:

> X2 capacity replacement is measured for **predictive dynamics**, not planning.

**Paper boundary (not a defect to patch):** X1/X2 parameter-efficiency is
now **decoupled from planning-efficiency**. Reachable-target feasibility
holds (oracle open-loop reconstruction \(\approx 0\)); failure is the
frozen CEM search budget on this host, not unreachable goals. Do **not**
enlarge the CEM grid to recover a planning claim.

## Harness notes

- Reachable \(q^\star\) via oracle open-loop torque rollout (feasibility by
  construction); open-loop reconstruction error \(\approx 0\).
- Success scored by best non-violating terminal distance (not cost-argmin
  alone).
- X1 planning gate remains untouched historical record (\(S_{\mathrm{plan}}=0\)).

## Unlock

\[
\text{P0 freeze complete}
\;\Rightarrow\;
\text{CAP-X2 }\rho\text{ capacity sweep allowed}
\]
