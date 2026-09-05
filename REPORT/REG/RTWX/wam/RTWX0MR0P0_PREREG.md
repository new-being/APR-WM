# RTWX-MR0-P0 预注册 — Action-Chunk Capability Qualification

日期：2026-09-01  
状态：**已冻结 / RAN**（`action_chunk_contract_failure`；`unlocks_mr0_a=false`；报告：`REPORT/REP/RTWX/wam/RTWX0MR0P0_REPORT.md`）  
依赖：MR0 STOP = `action_chunk_contract_failure`（\(H_a=0\)）  
**禁止**：repeat 单步动作；连续 8 次单步再拼接；hold-last-qpos；trajectory 插值；TASK-X1 diffusion；改 perception；看完结果再改 G3 阈值

## 唯一问题

能否得到一次 forward 即输出

\[
A_t=(a_t,\ldots,a_{t+H_a-1}),\qquad H_a\ge 8
\]

且任务能力足够的 **frozen** chunk policy？

## 合同

`ActionChunkSpec`：`horizon>=8`，`semantics=joint_position`，`policy_dt_env_ticks=1`，`deterministic=true`。  
dt≠1 → FAIL（禁止插值）。

## 门

| Gate | 失败 pattern |
|------|----------------|
| G0 接口 \(H_a\ge8\) / loader | `action_chunk_contract_failure` |
| G1 时间轴来自模型，非 adapter repeat | `action_chunk_contract_failure` |
| G2 native buffer 执行，无 hidden replan | `action_chunk_execution_failure` |
| G3 \(K_a=1\) receding，pooled success \(\ge 0.40\)，单任务 \(\ge 0.20\)，safety \(\le 0.20\)；\(N=32\)；任务=`place_empty_cup`,`put_object_cabinet`,`stamp_seal` | `action_chunk_policy_incompetent` |

全部通过：`action_chunk_qualified` → 才允许 MR0-A。

G3 阈值在本文件冻结，**不是** MR0-A 看完再定。

## 策略来源

- 情况 A：已有 chunk checkpoint → 冻结 sha256。  
- 情况 B：训练 = **另开 provisioning 格**，不是本格 science。本格 **不训练**。TASK-X1 仍 LOCKED。

## 实现

`aprwm_v0/rtwx_mr0_p0.py`；`control/action_chunk_spec.py`；`control/chunk_policy_adapter.py`；复用 `ActionChunkBuffer`
