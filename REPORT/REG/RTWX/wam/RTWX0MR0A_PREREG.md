# RTWX-MR0-A 预注册 — Action Resampling Sweep

日期：2026-09-01  
状态：**已冻结 / RAN-BLOCKED**（`mr0_a_blocked_unqualified_policy`；报告：`REPORT/REP/RTWX/wam/RTWX0MR0A_REPORT.md`）  
依赖：[`RTWX0MR0P0_PREREG.md`](RTWX0MR0P0_PREREG.md)  
**禁止**：\(K_s\) sweep；strided ICP；十字格；adaptive replan；hold-last；插值；改 G3/non-inferiority 阈值；改任务或 \(N\)

## 问题（只验证 \(H_A\)）

\[
H_A:\ \text{action plan 在多个 control tick 内是否有 temporal reuse 价值？}
\]

**不**声称 \(f_s>f_a\)。\(H_S\) 另格，且当前 G0-S 不合格。

## 矩阵

\(K_s=1\) 冻结。\(K_a\in\{1,2,4,8\}\)。四格。  
任务 / \(N=50\) / CRN seeds（38601 起）/ non-inferiority（pooled 5pp，单任务 10pp，safety 2pp）**继承 MR0**。

总量 \(3\times50\times4=600\)。

## Pattern

\[
K_a^*=\max\{K_a:\text{non-inferior vs }K_a=1\}
\]

- \(K_a^*\ge4\) → `action_temporal_reuse_supported`
- \(K_a^*=2\) → `action_temporal_reuse_limited`
- \(K_a^*=1\) → `action_temporal_reuse_not_supported`

P0 未 qualified → `mr0_a_blocked_unqualified_policy`（本格不进入 science）。

## 实现

`aprwm_v0/rtwx_mr0_a.py`；`evaluation/action_resampling_metrics.py`
