# RTWX-AC3 报告 — Demonstration Execution Contract Audit

日期：2026-09-01  
状态：**RAN / STOP at P0**  
pattern = `native_expert_replay_failure`  
`environment_replay_contract_failure` = **true**  
预注册：`REPORT/REG/RTWX/wam/RTWX0AC3_PREREG.md`  
产物：`runs/rtwx_ac3/`  
**未**训 policy。**未**跑 R0（B1–B3）。

## 一句话

\[
\boxed{\texttt{native\_expert\_replay\_failure}}
\]

官方 native 路径要求的 `_traj_data/episode{i}.pkl` **全部缺失**（held-out 24/24）。因此还不能把 AC2 的 1/24 归因于 velocity 或 downsampling——**连 expert 自己的 dense path 都无法加载**。

D0 在现有 converted HDF5 上仍查出两件硬事实：标签是 **next-index**；**没有任何 \(\dot q^{des}\)**。

## P0 Native replay

合同：同 seed → `load_tran_data` → `need_plan=False` → `set_path_lst` → `play_once` → `check_success`。  
\(N=8\)/task，门 7/8。

轨迹文件：`data/demo_clean/{task}/aloha_agilex/_traj_data/episode{idx}.pkl`  
三任务 held-out 共 24 个：**0 存在**。P0 全部 `err=missing_traj_data`，pooled=0，未进仿真。R0 正确跳过。

本机只有 converted 包：`data/demo_clean/.../data/episode_XXXXXXX.hdf5`（`state`/`action`）。attrs 写 `source_path=data/{task}/demo_clean/data/episode0.hdf5`，该 raw 文件也不在仓库。`frequency=15`（= `save_freq`）写在 additional_info 里，但没有 dense path。

## D0（文件审计；Q1/Q2 因无 native trace 未闭合）

| 问 | 结果 |
|----|------|
| Q1 `joint_action` ≟ \(q^{des}_{k_j}\) | **未测**：无 raw HDF5、无 dense trace |
| Q2 相邻 frame 的 dense 间隔 \(r\) | **未测**：无 `saved_frame_idx` |
| Q3 \(\dot q^{des}\) | **确认缺失**。converted 全树无 qvel/velocity 键。`Q3_velocity_absent_all=true` |
| Q4 same vs next | **next_index**。三任务 \(e_{\rm next}=0\)，\(e_{\rm same}\approx0.005\) |

Q4 不是猜 converter 默认值。数值上：

\[
a_t^{\rm HDF5}=s_{t+1}^{\rm HDF5},\qquad e_{\rm next}=0.
\]

AC0 cache 的 `a` 与 converted `action` 的 \(e_0=0\)：训练标签就是这套 **next-frame action**。

## 解释边界

当前能写的：

1. Native expert 重放合同在本数据发行版上**不可执行**（缺 `_traj_data`）。  
2. 现用 HDF5 **没有** planner velocity target。  
3. AC0/X1 学的是 \(s_t\to\) 下一帧 drive-target 记录，不是同 index 的 \(q^{des}_t\)。

还**不能**在 R0 上区分：

- missing \(\dot q^{des}\)  
- sparse vs dense timing  
- 仅 alignment  

那需要先拿到 `_traj_data`（或官方 raw `joint_action` HDF5）再跑 B1–B3。

```text
R0_run = false
```

下一格：恢复 RoboTwin 采集产物（`_traj_data`），不要再训 policy。
