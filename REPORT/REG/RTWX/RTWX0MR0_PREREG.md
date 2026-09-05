# RTWX-MR0 预注册 — Multi-Rate State–Action Resampling Probe

日期：2026-09-01  
状态：**已冻结 / RAN**（`action_chunk_contract_failure`；报告：`REPORT/REP/RTWX/wam/RTWX0MR0_REPORT.md`）  
依赖：MEMB0 `winner=none`（停止 WP 局部 DFS）；SEQ0/BEL0 relative backbone；冻结 ICP  
**禁止**：4×4 网格；hold-last-qpos 假装 \(K_a>1\)；dynamics extrapolation；改 ICP / policy 权重 / env 控制频率；看完结果再换任务或加 \(N\)

## 问题

\[
\text{task success 对 world-state staleness 是否比对 action-plan staleness 更敏感？}
\]

若成立，支持 \(f_{\mathrm{state}}\gg f_{\mathrm{action-replan}}\)（底层控制 Hz 不变）。

## 合同

- \(s^R=(q,\dot q)\) **每 tick 新鲜**；\(K_s\) 只陈旧 \(s^O\)。
- \(K_a>1\)：**消费 action chunk**，不是降低 actuator 频率。
- G0-A：\(H_a\ge 8\)。否则 pattern = `action_chunk_contract_failure`，**停止**，禁止 target-hold。
- G0-S：stride ICP \(P_t\leftrightarrow P_{t+K_s}\)；\(K_s=8\) valid rate \(<0.8\) → 该格 instrument unsupported。
- Compute-skipping：更新 tick 做 **一次** multi-step registration，不跑 adjacent 补链。不 extrapolate。
- 十字 7 cell：B0 \((1,1)\)；S2/S4/S8；A2/A4/A8。禁止 \((4,4)\) 等。
- 任务写死：`place_empty_cup`（transport）、`put_object_cabinet`（contact）、`stamp_seal`（precision）。
- \(N=50\) paired CRN；CLI 不开放 strides/tasks/阈值。

## Pattern（先 joint）

```text
if Ks*>=4 and Ka*>=4: joint_low_rate_supported
elif Ka* >= 2*Ks*:   multirate_asymmetry_supported
else:                multirate_asymmetry_not_supported
```

Non-inferiority（相对 B0）：pooled success drop \(\le 5\) pp；单任务 \(\le 10\) pp；safety 增 \(\le 2\) pp。

Primary = task success。不合成加权 \(J\)。

## 实现

`aprwm_v0/rtwx_mr0.py`；`control/multirate_scheduler.py`；`control/action_chunk_buffer.py`；`evaluation/multirate_metrics.py`
