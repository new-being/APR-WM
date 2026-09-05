# RTWX-TASK-X1-I1 报告 — Diffusion Parameterization Breadth Probe

日期：2026-09-01  
状态：**RAN**  
pattern = `epsilon_parameterization_pathology_supported`  
winner = **B2** (`sample` / 直接 \(x_0\)-prediction)  
预注册：`REPORT/REG/RTWX/wam/RTWX0TASKX1I1_PREREG.md`  
产物：`runs/rtwx_task_x1_i1/`

## 一句话

\[
\boxed{
\epsilon\text{-prediction 在 }t=99\text{ 仍爆（}L1_{\rm norm}\approx558\text{）；}
v\text{ 与 }x_0\text{ 都把该点压到 }<0.13\text{，且 Stage A G0--G4 全 PASS。}
}
\]

\[
\boxed{
\text{失败主因是 }\epsilon\text{ 低 SNR 反演病态，不是 “diffusion 训不好”。}
}
\]

不声称 v 一定更好：两支都合格时，按预注册先比 \(L1_{\rm norm}(99)\)，B2 明显更低，故 winner = \(x_0\)。

## 冻结合同

唯一 \(\Delta\) = `prediction_type`。AC0 split、\(s_t,z_g\)、\(H_a=8\)、\(d_a=14\)、z-score、256×3 MLP（\(n_\theta=247968\)）、cosine \(K=100\)、10-step DDIM、\(\eta=0\)、AdamW、**满 100 epoch（无 early stop）**、无 clip / process / history / Min-SNR / truncated schedule。**未**进 52601 闭环。

B0 **不重训**，同 I0 第一批 val chunk + seed-0 噪声。

训练损失 \(L_\epsilon,L_v,L_{x_0}\) **不可比**，未用于选 winner。ckpt 只按各自 val loss 存盘。

## Reconstruction（\(x_0\) 空间，\(t\in\{10,50,99\}\)）

门槛：G0 \(L1(10)\le0.15,\ L1(50)\le0.30\)；G1 \(L1(99)\le2.0\)；G2 \(D_{\rm pred}(99)\le 10 D_{\rm demo}\)。

| 支 | 类型 | \(L1(10)\) | \(L1(50)\) | \(L1(99)\) | \(D_{\rm pred}(99)\) | rec |
|----|------|------------|------------|------------|---------------------|-----|
| B0 | epsilon（冻结 X1） | 0.091 | 0.193 | **558** | 1966 vs demo 0.047 | FAIL G1/G2 |
| B1 | v-prediction | 0.065 | 0.155 | **0.126** | 0.382 | PASS |
| B2 | sample / \(x_0\) | 0.021 | 0.023 | **0.026** | 0.060 | PASS |

B0 与 I0-C2 数值重合（同 batch、同 \(t=99\)）。把 558 压到 0.13 / 0.026 是 **>3 个数量级**，超过 G1 的「两个数量级」资格门。

## Stage A（仅 rec PASS 后；完整 10-step DDIM）

| 支 | G0–G4 | raw L1 | \(e_0\) | \(e_7\) | \(D_{\rm pred}/D_{\rm demo}\) |
|----|-------|--------|---------|---------|------------------------------|
| B0 | skipped | — | — | — | — |
| B1 | **PASS** | 0.062 | 0.058 | 0.072 | 0.368 / 0.043 |
| B2 | **PASS** | 0.014 | 0.011 | 0.021 | 0.052 / 0.043 |

B2 离线误差与 AC0-B0 点回归同量级。B1 多样度偏高但仍过 G4（下界 0.25×demo）。

## Winner

两支 Stage A 全 PASS。\(L1_{\rm norm}(99)\): 0.126 vs 0.026，相对差 \(\gg 10\%\)，**不是** complexity tie，不偏向 v。

```text
winner = B2
prediction_type = sample
ckpt = runs/rtwx_task_x1_i1/B2/best.pt
```

## Pattern

```text
epsilon_parameterization_pathology_supported
```

两个不含 \(1/\alpha_t\) 病态反演的目标都修好了，冻结 \(\epsilon\) 仍只在 \(t=99\) 爆。这比「v 碰巧更好」更强：高噪声稳定性主要来自**去掉 \(\epsilon\) 反演放大**。

本格 **不** 跑 Stage B（`unlocks_stage_b=false`）。仪器已合法：`qualified_for_x1_stage_b=true`。下一步才是用 B2 ckpt 做原 X1 Stage B bakeoff（seeds **52601**，门不变）。不要在本格改 Stage B。
