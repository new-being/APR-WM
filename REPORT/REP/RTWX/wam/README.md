# wam

多速率 / 调度层（不是 world-perception DFS）。

- **RTWX-AC0**：**RAN / `covariate_shift_suspected`** — [`RTWX0AC0_REPORT.md`](RTWX0AC0_REPORT.md)
- **RTWX-AC1-D0**：**RAN / `state_support_metric_invalid`** — [`RTWX0AC1D0_REPORT.md`](RTWX0AC1D0_REPORT.md)
- **RTWX-AC1-D1**：**RAN / `support_geometry_unqualified`** — [`RTWX0AC1D1_REPORT.md`](RTWX0AC1D1_REPORT.md)
- **RTWX-AC2**：**RAN / `expert_action_replay_failure`** — [`RTWX0AC2_REPORT.md`](RTWX0AC2_REPORT.md)
- **RTWX-AC3**：**RAN / `native_expert_replay_failure`** — [`RTWX0AC3_REPORT.md`](RTWX0AC3_REPORT.md)
- **RTWX-MA0**：**RAN / `official_act_stack_unqualified`（cup 3/16，G0 FAIL；非零闭环已见）** — [`RTWX0MA0_REPORT.md`](RTWX0MA0_REPORT.md)
- **RTWX-MR0**：**RAN / `action_chunk_contract_failure`** — [`RTWX0MR0_REPORT.md`](RTWX0MR0_REPORT.md)
- **RTWX-MR0-P0**：**RAN / `action_chunk_contract_failure`** — [`RTWX0MR0P0_REPORT.md`](RTWX0MR0P0_REPORT.md)（权重已拷；**未**重跑 G3）
- **RTWX-MR0-A**：**BLOCKED** — [`RTWX0MR0A_REPORT.md`](RTWX0MR0A_REPORT.md)
- **RTWX-TASK-X1**：**RAN / Stage A `diffusion_chunk_instrument_failure`；Stage B `diffusion_chunk_breadth_insufficient`** — [`RTWX0TASKX1_REPORT.md`](RTWX0TASKX1_REPORT.md) · [`RTWX0TASKX1_STAGEB_REPORT.md`](RTWX0TASKX1_STAGEB_REPORT.md)
- **RTWX-TASK-X1-I0**：**RAN / `diffusion_denoiser_insufficient`** — [`RTWX0TASKX1I0_REPORT.md`](RTWX0TASKX1I0_REPORT.md)
- **RTWX-TASK-X1-I1**：**RAN / `epsilon_parameterization_pathology_supported`** — [`RTWX0TASKX1I1_REPORT.md`](RTWX0TASKX1I1_REPORT.md)

NEXT：MA0-P1 cup **3/16 < 4/16**，C0/MA1 锁定。官方栈已证明非零闭环；先对齐参考环境/数据，不要改 G0 凑过。
