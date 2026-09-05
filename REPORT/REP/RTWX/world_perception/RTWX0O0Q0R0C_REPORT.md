# RTWX-O0Q0R0C 报告 — Native Event-Trace & Excitation Qualification

日期：2026-08-30  
状态：**正式冻结** `native_demo_action_replay_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0C_PREREG.md`  
产物：`runs/rtwx_o0q0r0c/{summary.json,run.json}`  
Seeds 0/1/2；无 IK；无接触 restore；无 23 格 yaw  
**O0 target 未改。** O0Q0R1 LOCKED。

## 一句话

P0/P1 仪器与 \(H=6\) 强惯量对照都过了；失败在 **官方 hdf5 qpos 当 `take_action` target 重放仍到不了杯子**。

\[
\boxed{\texttt{native\_demo\_action\_replay\_failure}}
\]

## Gates

| Gate | 结果 |
|------|------|
| G0-regression | **PASS**（P0 \(D_H(I)\le 7.3\times10^{-4}\)，无落地；P1 \(\sim 2.6\times10^{-5}\)） |
| G1-native-event | **FAIL**（P2=0，P3=0，\(\min d\sim 0.21\)–\(0.27\)） |
| G2-branch-repro | **FAIL**（无 event，无 suffix） |
| G3-P0-excite | **PASS**（\(I_x/I_z=4\)：\(D_H(R_y90)=0.68/0.48/0.48>\tau\)） |
| G4-P1-excite | **PASS**（摩擦 \(0.16>\tau\)） |
| G5-P2/P3-excite | **FAIL**（无接触样本） |

## 主表

| 项 | 值 |
|----|-----|
| \(H_{P0}\) | **6**（未改） |
| P0 \(D_H(I)\) | 0.00073 / \(1.8\times10^{-5}\) / \(1.8\times10^{-5}\)；接触 **false** |
| P1 \(D_H(I)\) | \(\sim 2.6\times10^{-5}\) |
| demo channel | **qa**（`take_action` qpos）；\|A\|=179/168/172 |
| \(\min d(\mathrm{EE},\mathrm{cup})\) | 0.263 / 0.205 / 0.268 |
| P0 probe \(I_x/I_z=4\) | **0.677 / 0.484 / 0.484** \(>\tau=0.05\) |
| P1 probe | **0.159 / 0.156 / 0.163** \(>\tau\) |

## 读数

1. **不要再加大 P0 control，也不要再动 \(H\)。** \(I_x/I_z=4\) 配非主轴 \(\omega=(2.0,0.3,1.5)\) 后，短窗已经能看见 yaw-dependent inertia。R0B 的 \(0.036<\tau\) 是激励强度问题，不是 horizon。
2. **P1 仍是 regression anchor。** identity 与摩擦对照都未回归。
3. **hdf5 qpos 不足以作为可重放 action。** 同 seed 官方 `action` 经 native `take_action` 重放，手臂停在桌旁，\(\min d\) 与 R0B 的 IK 失败同一量级。demo 只提供了动作通道，没有把机器人送进 P2/P3。
4. **事后 hdf5 审计（非门）：** `action` qpos 与下一帧 `state` **逐元素相同**（\(\|q^{a}_t-q_{t+1}\|=0\)）。这是移位观测，不是独立的 low-level command。R0 用 `set_arm_joints` 灌观测 q 能看见接触态，那是 restore，不是前向 replay。
5. **O0Q0R1 仍 LOCKED。** 下一步是审计 recorder / controller 的真实 action channel（dense / EE / 实际写入 `take_action` 的量），不是第三套 planner，也不是 23 yaw。

## Pattern

```text
pattern = native_demo_action_replay_failure
G0 = PASS
G1 = FAIL (P2=0, P3=0, min_d~0.21–0.27)
G2 = FAIL
G3 = PASS (P0 I=4 D_H~0.48–0.68)
G4 = PASS (P1 fric~0.16)
G5 = FAIL (no contact)
unlocks_o0q0r1_prereg = false
o0_target_remains_full_TR = true
```
