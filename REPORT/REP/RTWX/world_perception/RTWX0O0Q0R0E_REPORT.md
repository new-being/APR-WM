# RTWX-O0Q0R0E 报告 — Native Dense-Trace Acquisition

日期：2026-08-30  
状态：**正式冻结** `dense_trace_unavailable`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0Q0R0E_PREREG.md`  
产物：`runs/rtwx_o0q0r0e/{summary.json,run.json}`  
Seeds 0/1/2；无新 planner；无 P0/P1 重开；无 23 yaw  
**O0 target 未改。** O0Q0R1 LOCKED。

## 一句话

本机两条允许路径都空：没有 CuRobo，官方 demo 也没有 `_traj_data`。完整 \(A^*_{\mathrm{raw}}\) 拿不到。

\[
\boxed{\texttt{dense\_trace\_unavailable}}
\]

## Gates

| Gate | 结果 |
|------|------|
| G0-provenance | **FAIL**（无 source） |
| G1-complete | **FAIL** |
| G2-replay | **FAIL** |
| G3-branch | **FAIL** |

## 路径探测

| 路径 | 结果 |
|------|------|
| A CuRobo collection | `ModuleNotFoundError: No module named 'curobo'` |
| B `_traj_data/episode{0,1,2}.pkl` | 不存在（`demo_clean/.../aloha_agilex/` 只有 hdf5/video/seed.txt） |

官方 `collect_data.py` 在 seed 阶段会 `save_traj_data`，但随 hdf5 发布的包**没有**带上这些 pkl。hdf5 仍只是 `frequency=15` 的抽样，不能代替 dense drive。

## 读数

1. 反事实方法（R0B 预接触分叉、R0C 激励、R0D 控制边界）**不是**当前主阻塞。
2. 阻塞是基础设施：要在**能跑通官方 `play_once` 的 CuRobo 环境**里挂 `set_drive_target` logger，或找回当时的 `left_joint_path`。
3. 本格没有用 mplib 顶替，也没有把 hdf5 再当 \(A^*\)。
4. **O0Q0R1 仍 LOCKED。**

## Pattern

```text
pattern = dense_trace_unavailable
path_A = FAIL (no curobo)
path_B = FAIL (no _traj_data pkl)
unlocks_o0q0r1_prereg = false
o0_target_remains_full_TR = true
```
