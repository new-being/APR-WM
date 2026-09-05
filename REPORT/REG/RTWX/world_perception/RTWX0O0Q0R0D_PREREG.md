# RTWX-O0Q0R0D 预注册 — Native Control-Channel Audit

日期：2026-08-30  
状态：**正式冻结** `capture_incomplete`  
依赖：O0Q0R0C=`native_demo_action_replay_failure`  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0R0D_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0R0D_REPORT.md)  
**禁止**：修 mplib / 再写 IK；把 hdf5 qpos 当 \(A^*\)；23 格 yaw；改 \(\tau=0.05\) / \(H_{P0}=6\) / \(I_x/I_z=4\)；restore 接触态；改 O0 target；开 O0Q0R1 / O0Q1 / O1。

## 唯一问题

> 官方 expert **成功执行时**真正写进控制器/执行器的 action 是什么？该 actuator-side trace 能否无 planner 重放出 P2/P3？

demo / hdf5 **只作对照，不提供 \(A^*\)**。不预先假设 `action_type=qpos`。

## 钩子顺序

\[
\texttt{take\_action input}
\rightarrow
\text{dense/interpolated }q^{\mathrm{tar}}
\rightarrow
\text{SAPIEN drive target}
\]

保存的是**执行轨迹**（planner 之后、PD drive 之前），不是策略输出。重放禁止再跑 planner。

## 支路

| 支路 | 问 |
|------|------|
| B0 | successful expert 的 control path |
| B1 | seed 0/1/2 运行时捕获 \(A^*_{\mathrm{raw}}\) |
| B2 | \(A^*\) 与 \(q_t,q_{t+1}\) 的语义（不要求不相等） |
| B3 | 同 seed exact-path replay（只执行捕获 command） |

## Gates

| Gate | 条件 |
|------|------|
| G0-channel | 唯一定位 actuator-side channel（预注册：SAPIEN `set_drive_target` 为最后确定性边界） |
| G1-capture | 成功 expert 的每个 physics step 都有 command；无 silent gap |
| G2-robot | 同 seed replay：\(\max_t e_q<\epsilon_q=\mathbf{0.05}\)，\(\max_t e_{EE}<\epsilon_{EE}=\mathbf{0.03}\) |
| G3-event | replay 上 \(n_{P2}\ge 3\) 且 \(n_{P3}\ge 3\) |
| G4-branch | 预接触根上 \(D_H(I)<0.02\)（R0B 合同；不 restore 接触） |

## Patterns（无 yaw）

| Pattern | 条件 |
|--------|------|
| `control_channel_unresolved` | ¬G0 |
| `capture_incomplete` | G0 ∧ ¬G1 |
| `robot_replay_failure` | G0–G1 ∧ ¬G2 |
| `task_event_replay_failure` | G0–G2 ∧ ¬G3 |
| `precontact_identity_failure` | G0–G3 ∧ ¬G4 |
| `native_control_trace_qualified` | G0–G4 |

仅最后一项解锁 **O0Q0R1**。

R0C 冻结量本格不改：\(H_{P0}=6\)，\(I_x/I_z=4\)，\(\omega=(2.0,0.3,1.5)\)，P1 摩擦 probe，\(\tau=0.05\)。
