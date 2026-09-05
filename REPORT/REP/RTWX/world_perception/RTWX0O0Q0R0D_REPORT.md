# RTWX-O0Q0R0D 报告 — Native Control-Channel Audit

日期：2026-08-30  
状态：**正式冻结** `capture_incomplete`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0D_PREREG.md`  
产物：`runs/rtwx_o0q0r0d/{summary.json,run.json,A_star_seed*.npz}`  
Seeds 0/1/2；无 hdf5-as-action；无 IK 补救；无 23 格 yaw  
**O0 target 未改。** O0Q0R1 LOCKED。

## 一句话

最后确定性边界已经定位为 **SAPIEN drive target**；失败在 **没有 CuRobo 就无法跑完官方 expert，完整 \(A^*_{\mathrm{raw}}\) 截不下来**。

\[
\boxed{\texttt{capture\_incomplete}}
\]

## Gates

| Gate | 结果 |
|------|------|
| G0-channel | **PASS**（`sapien_joint_drive_target`；`take_action` 调用 **0** 次） |
| G1-capture | **FAIL**（成功 expert 0/3；仅 close-gripper 前缀 300 step） |
| G2-robot | **FAIL**（无完整 trace；前缀 \(e_{EE}\sim10^{-7}\)，\(e_q=0.4\) 为爪编码差） |
| G3-event | **FAIL**（P2=P3=0） |
| G4-branch | **FAIL**（无接触前根） |

## B0 — provenance

运行时观察到的 expert 路径：

```text
play_once
 -> close_gripper / take_dense_action
 -> robot.set_gripper -> set_drive_target -> scene.step   (300 steps)
 -> grasp_actor -> EE target -> plan_batch (CuRobo)
 -> RuntimeError: cuRobo is not installed
```

`take_action` **不在** expert 路径上。它是 eval 接口。  
hdf5 的 `joint_action` 来自 `get_*_jointState()`（臂关节 = `get_drive_target()`），且 `save_freq=15`，不是 250 Hz 执行轨迹。

## B2 — 语义（前缀，非完整 expert）

| 量 | 值 |
|----|-----|
| \(\|A^*-q_t\|\) mean/max | 1.19 / 1.35 |
| \(\|A^*-q_{t+1}\|\) | 1.19 / 1.35 |
| \(A^*\equiv q_{t+1}\) | **false** |
| `jointState` ≡ raw drive | **false**（臂=drive，爪是缓存的归一化开度） |

R0C 的 \(q^a_t\equiv q_{t+1}\) 是 **save-rate 下两次 drive 采样相邻**，不是 “action = next measured q”。

## 读数

1. **要重放的是 planner 之后的 dense drive trace**，不是 hdf5 qpos，也不是 `take_action(qpos)`。
2. **本机截不下完整 \(A^*\)。** 官方 expert 的臂运动依赖 CuRobo `plan_batch`；当前环境没有 curobo，`play_once` 在首次 grasp pose 选择处中断。禁止用 mplib 顶替去“造一条能抓的轨迹”。
3. 已截到的 300-step 爪指令可以按 drive 重放（EE 几乎不动，\(e_{EE}\sim10^{-7}\)），但这不是 P2/P3。
4. **O0Q0R1 仍 LOCKED。** 下一格需要在**同一条官方 expert 控制链**上拿到完整 actuator-side \(A^*\)（有 CuRobo 的 collection，或官方保存的 `left_joint_path`），而不是再猜 hdf5。

## Pattern

```text
pattern = capture_incomplete
G0 = PASS (sapien_joint_drive_target; take_action=0)
G1 = FAIL (play_once 0/3; CuRobo missing)
G2 = FAIL (no complete A*)
G3 = FAIL (P2=P3=0)
G4 = FAIL
unlocks_o0q0r1_prereg = false
o0_target_remains_full_TR = true
```
