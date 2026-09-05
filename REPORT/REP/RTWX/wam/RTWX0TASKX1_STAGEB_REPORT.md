# RTWX-TASK-X1 Stage B 报告 — Architecture Bakeoff（合法 \(x_0\)-diffusion vs AC0-B0）

日期：2026-09-01  
状态：**RAN**  
pattern = `diffusion_chunk_breadth_insufficient`  
winner = **none**  
预注册：`REPORT/REG/RTWX/wam/RTWX0TASKX1_PREREG.md`  
产物：`runs/rtwx_task_x1_sb/`（不覆盖 \(\epsilon\) 仪器失败的 `runs/rtwx_task_x1/`）  
checkpoint = `runs/rtwx_task_x1_i1/B2/best.pt`（`prediction_type=sample`）  
**未**写入 `runs/rtwx_mr0_p0/`。**未**跑 C0。**未**插 I2。

## 一句话

\[
\boxed{
S_{\rm pooled}^{\rm B0}=0,\qquad S_{\rm pooled}^{\rm B2}=0
}
\]

合法的 direct-\(x_0\) diffusion chunk 在 52601 CRN bakeoff 上与已经 0-success 的确定性回归**同为零**。这是有仪器的负结果，不是 Stage A 爆炸。

## 测的是什么（claim 边界）

本格只回答：

\[
\boxed{\text{action-model family change 有无闭环效用}}
\]

**不能**写成 “multimodal diffusion beats regression”。B1 是 \(x_0\)-prediction，离线 \(L1\approx0.014\)，已接近点回归；即使它曾成功，原因也可以是条件去噪训练 / 迭代细化 / 噪声增强，而不是多模态。本次两边都是 0，连 family 效用都没有。

## 冻结合同

| 项 | 值 |
|----|-----|
| B0 | 冻结 AC0 确定性 MLP |
| B1 | I1-B2 direct \(x_0\) diffusion chunk |
| 状态 | oracle \(s_t,z_g\) |
| chunk | \(H_a=8,\ d_a=14,\ K_a=1\)，每 tick 重生成，只执行 \(A_t[0]\) |
| split | 同 AC0；无 process / history / 新数据 |
| seeds | **52601**–52616，\(N=16\)/task，各 48 幕 CRN |
| Stage B 门 | 原样：\(S_{\rm pooled}\ge0.20\)、\(\Delta\ge0.15\)、至少两任务 success>0、\(\Delta\)safety\(\le0.02\) |

Stage A 已在 I1-B2 闭合（G0–G4 PASS，raw L1 0.014，\(e_0/e_7\approx0.011/0.021\)，\(D_{\rm pred}\approx D_{\rm demo}\)）。本格不再训、不改 parameterization。

## 结果

| | pooled | safety | cup | cabinet | seal | \(n_{>0}\) |
|--|--------|--------|-----|---------|------|------------|
| B0 MLP | **0/48** | 0 | 0 | 0 | 0 | 0 |
| B1 \(x_0\)-diff | **0/48** | 0 | 0 | 0 | 0 | 0 |

96/96 幕跑满（ticks 400–700，无 worker err、无 safety）。\(\Delta=0\)。门全 FAIL。

```text
pattern = diffusion_chunk_breadth_insufficient
winner = none
unlocks_c0 = false
unlocks_mr0_p0 = false
stop_diffusion_local_dfs = true
```

## 解释

仪器已经健康：不是 \(\epsilon\) 反演病态，不是 sampler 合同错误。闭环仍与 AC0-B0 一样 0-success，说明在当前 oracle 状态、专家数据、\(K_a=1\) chunk 合同下，**把点回归换成生成式 \(x_0\)-chunk 没有产生闭环效用**。

按预注册：停止 diffusion 本地 DFS。不扫 v、denoise steps、Transformer、Min-SNR、更大网络。不进 TASK-X1-C0，不把权重塞回旧 single-forward MR0-P0。
