# R1-MJ/RS — MJ0 + RS0 Closure Report

Date: 2026-08-15

## Decision

R1-MS0 (ManiSkill Drawer) remains:

```text
frozen = true
status = infrastructure_block
scientific_no_go = false
```

The active realism bridge is:

```text
R1-MJ0 → R1-RS0 → R1-RS1 → R1-RS2
```

This report covers the first two infrastructure gates only. It is **not** an
R0.6 revision-transfer scientific result.

## Packages (frozen for this bridge)

| package   | version |
|-----------|---------|
| MuJoCo    | 3.11.0  |
| robosuite | 1.5.2   |

MuJoCo 3.11 removed `mjData.qM`. APR-WM installs a robosuite-compatible
`mj_fullM` / `qM` shim and never uses the deleted storage in learner code.

## R1-MJ0 — native single-hinge force closure

- Matrix: seeds `{8201,8211,8221}` × trajectories `{sine,chirp,piecewise}` = 9 cells
- Duration: 10 s, `dt=0.002`
- Gate: every cell `NRMSE_τ < 1e-4`
- Result: **passed**
- Max cell NRMSE: `8.21e-11`
- Snapshot restore / deterministic replay errors: `0`

Outputs: `runs/r1_mj0/closure/`

## R1-RS0 — robosuite Door Mode-A C0

Mode A: Door + Panda present, robot kinematically frozen, latch off, torque on
`Door_hinge` only. Fluid viscosity/density disabled for hinge closure.

Physics profiles (adequate C0; learner knows matching damping):

| Profile | frictionloss | damping |
|---------|-------------:|--------:|
| P0      | 0.10         | 0.10    |
| P1      | 0.05         | 0.05    |
| P2      | 0.20         | 0.20    |

Important MuJoCo accounting fact used here:

- viscous `dof_damping` → `qfrc_passive`
- `frictionloss` → absorbed in `qfrc_constraint`

Learner residual therefore models damping only and keeps
`include_constraint=True`. Truth `qfrc_passive` is logged under `raw_truth`
only and never enters `learner_visible`.

- Matrix: profiles `{P0,P1,P2}` × seeds `{8301,8311,8321}` = 9 cells
- Duration: 10 s
- Gates: modeled `NRMSE_τ < 1e-3`, false revision ≤1%, H32 validity ≥95%,
  all cells individually close
- Result: **passed**
- Max cell NRMSE: `2.79e-4`
- Unlocks RS1 eligibility; RS2 / Wipe / ToolHang remain locked

Outputs: `runs/r1_rs0/c0/`

## What this does / does not claim

Does:

- prove MuJoCo force-space adapter closure on a minimal hinge;
- prove robosuite Door hinge Mode-A closure across damping/friction scales;
- preserve R1-MS0 as an infrastructure block, not a scientific failure.

Does not:

- transfer R0.6 revision science (that is RS1);
- test robot-contact revision (RS2);
- train policies or use vision.

## Next unlocked experiment

```text
R1-RS1: Frozen R0.6 under Door Mode-A misspecification
        (C0 / C1-L / C1-H / CNEG / C2-latch)
```
