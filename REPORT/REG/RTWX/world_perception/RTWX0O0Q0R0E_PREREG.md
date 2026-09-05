# RTWX-O0Q0R0E 预注册 — Native Dense-Trace Acquisition

日期：2026-08-30  
状态：**正式冻结** `dense_trace_unavailable`  
依赖：O0Q0R0D=`capture_incomplete`  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0R0E_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0R0E_REPORT.md)  
**禁止**：新 planner / mplib 顶替；hdf5 当 \(A^*\)；23 yaw；重开 P0/P1；改 \(\tau=0.05\)；restore 接触；改 O0 target；开 O0Q0R1 / O0Q1 / O1。

## 唯一问题

> 能否从**同一条官方 CuRobo expert 链**取得每个 physics step 的 \(q^{\mathrm{drive}}_t\)，并在去掉 planner 后重放出 P2/P3？

## 允许路径

| 路径 | 条件 |
|------|------|
| A | 本机 `curobo` 可 import；`play_once` 成功；logger 只挂 `set_drive_target`，不改 controller |
| B | 官方 `_traj_data/episode*.pkl` 的 `left_joint_path` 经审计确为 `take_dense_action` 消费的 dense sequence |

禁止：sparse waypoints、观测 qpos、`save_freq=15` 降采样。

## Gates

| Gate | 条件 |
|------|------|
| G0-provenance | 完整 trace 可追溯 `plan_batch`/`left_joint_path` → `take_dense_action` → `set_drive_target`；且有臂 drive（不能只是爪前缀） |
| G1-complete | 成功 expert：每 step 有 \(q^{\mathrm{drive}}_t\)；\(n_{\mathrm{step}}>400\)；出现 P2（三 seed） |
| G2-replay | 去掉 planner，同 seed 只放 \(A^*_{\mathrm{raw}}\)：\(\max e_q<0.05\)，\(\max e_{EE}<0.03\)；\(n_{P2}\ge3\)，\(n_{P3}\ge3\) |
| G3-branch | 预接触根 \(D_H(I)<0.02\) |

## Patterns

| Pattern | 条件 |
|--------|------|
| `dense_trace_unavailable` | 无路径 A/B，或 ¬G0 |
| `dense_trace_semantics_failure` | 有候选但 ¬G1（稀疏 / 非 dense / 无臂） |
| `native_trace_replay_failure` | G0 ∧ G1 ∧ ¬(G2 ∧ G3) |
| `native_dense_trace_qualified` | G0–G3 |

仅最后一项解锁 **O0Q0R1**。

P0 \(H=6\)、\(I_x/I_z=4\)、P1 摩擦 probe **本格不跑**。
