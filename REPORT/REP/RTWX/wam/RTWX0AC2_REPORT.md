# RTWX-AC2 报告 — Policy-State Sufficiency Probe

日期：2026-09-01  
状态：**RAN / STOP at P0**  
pattern = `expert_action_replay_failure`  
winner = **none**（未训练）  
预注册：`REPORT/REG/RTWX/wam/RTWX0AC2_PREREG.md`  
产物：`runs/rtwx_ac2/P0/`  
**未**训 B0/B1/B2，**未**进 Stage B，**未**写入 MR0-P0。

## 一句话

\[
\boxed{
\texttt{expert\_action\_replay\_failure}
}
\]

held-out 专家 `qpos` 动作按原 demo seed 逐 tick 重放，**不能**在当前仿真执行栈上复现示范成功。因此讨论 \(s_t\) 是否充分、或 \(H_a=8\) 是否多余，**尚未有资格**。

## P0 合同

- 数据：AC0 `split_manifest` 的 val+test 前 8 幕/任务（held-out，未进训练）
- 恢复：原 `seed.txt` + 同 index hdf5
- 动作：hdf5 `action` joint_position，`take_action(..., action_type="qpos")`，与 AC0 标签语义相同
- \(N=8\)/task，共 24 幕；门 \(S^{\rm each}\ge 0.875\)（7/8）

## 结果

| 任务 | success | safety | 备注 |
|------|---------|--------|------|
| place_empty_cup | **0/8** | 0 | 跑满 165–175 步 |
| put_object_cabinet | **0/8** | 0 | 跑满 263–278 步 |
| stamp_seal | **1/8** | 0 | 仅 ep=1 seed=1 |

pooled = 1/24 ≈ 0.042。无 worker 错误、无 TOPP/timeout safety：不是环境崩了，是 **replay 完了但 `check_success` 几乎全假**。

```text
training_run = false
unlocks_mr0 = false
```

## 这意味着什么

dataset action → current simulator execution **合同未闭合**。  
AC0/X1 的 offline-good / closed-loop-0 现在多了一个更硬的前置解释：专家动作本身在当前 control mode / timestep / TOPP 下就不能把同一初始状态带到成功。

按预注册 **停训**。不进入 B0/B1/B2，不加长 history，不上 RNN。下一格应查 execution semantics（动作通道、时间尺度、是否需要与录制时相同的控制器），而不是再换一个 policy 头。
