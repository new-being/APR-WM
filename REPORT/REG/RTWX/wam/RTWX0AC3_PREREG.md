# RTWX-AC3 预注册 — Demonstration Execution Contract Audit

日期：2026-09-01  
状态：**已冻结**  
依赖：AC0 split held-out 8/task；**禁止**训练 policy；禁止一上来扫 alignment \(\Delta\)

## 问题

RoboTwin native expert control 是在哪一步被压成不可重放的 HDF5 action？

## P0 Native expert replay

同 demo seed；加载 `_traj_data`；`need_plan=False`；`set_path_lst`；`play_once()`；`check_success()`。  
\(N=8\)/task。门 \(S^{\rm each}\ge 7/8\)。  
FAIL → `native_expert_replay_failure`，STOP（含缺失 traj / 版本 / seed / asset）。

PASS 则记录 dense control trace：\(q^{des},\dot q^{des},q^{actual}\)，以及 `_take_picture` 的 frame↔sim_step。

## D0 Provenance（不执行新动作）

Q1：`joint_action` 是否等于 \(q^{des}_{k_j}\)  
Q2：相邻 HDF5 frame 的 dense step 间隔 \(median/P90/\max(r)\)  
Q3：数据集是否保存 \(\dot q^{des}\)  
Q4：AC0 action 是 same-index 还是 next-index（数值，不靠代码记忆）

## R0（仅 native PASS）

- B0：AC2 sparse HDF5 `take_action(qpos)`（已知 ~1/24）
- B1：dense \((q^{des},\dot q^{des})\) 逐步 `set_arm_joints`（阳性对照）
- B2：同 dense 时序，\(\dot q^{des}=0\)
- B3：HDF5 \(q^{des}\) 按真实 \(r_j\) hold

Patterns：`action_state_missing_velocity_target` / `action_timestep_contract_failure` / `action_alignment_failure`（后者仅当 dense PASS 且 cadence 仍 FAIL 后才允许 \(\Delta\in\{-1,0,+1\}\)）。
