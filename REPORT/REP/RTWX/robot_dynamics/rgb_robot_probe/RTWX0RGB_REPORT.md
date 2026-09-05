# RTWX-X0RGB 报告 — Sim RGB → State → Frozen M2

日期：2026-08-29  
状态：**正式冻结** `perception_failure`  
预注册：`REPORT/REG/RTWX/RTWX0RGB_PREREG.md`  
产物：`runs/rtwx_x0rgb/{run.json,metrics.json,cache_rgb_splits.npz,W_m2.npy}`  
任务：`put_object_cabinet`；seed **14601**；48/24/24 × 120；`front_camera` 64×64；\(L=2\)。  
部署合同：\(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)。  
**不声称真实相机**；不改 X0E–X0EH2 ledger；TASK-XL / R10 仍 LOCKED。

## 一句话

冻结 M2 在 **oracle state** 上仍 competent（B0 \(E_{\mathrm{exec}}^{K=4}=0.653\)，\(r_{\mathrm{cat}}=0\)）。  
从仿真 RGB 恢复显式关节状态 **失败**：\(E_q=1.018\)，\(E_{\dot q}=1.029\)，远超 \(0.35/0.50\)。  
Primary pattern：**`perception_failure`**。本格 **不能** 声称 84× 结构优势穿过了视觉接口。

## 输入 / 输出 / 评价（本格）

| 量 | 含义 |
|----|------|
| \(s_t=(q_t,\dot q_t)\) | oracle 关节状态（真值） |
| \(a_t=q_t^{\mathrm{tar}}\) | oracle qpos target（条件，不评误差） |
| \(RGB_{t-1:t}\) | `front_camera` 64×64 |
| \(E_{\mathrm{vis}}\) | \(RGB_{t-L:t}\to(\hat q,\hat{\dot q})\) |
| 误差 | 只评 **状态预测**；不评 action |

NRMSE 同 `aprwm_v0/rtwx_x0._nrmse`：\(\mathrm{RMSE}(\hat y,y)/(\mathrm{RMS}(y)+\epsilon)\)。

## Primary gates

| Gate | 条件 | 结果 |
|------|------|------|
| G_oracle | B0 \(r_{\mathrm{cat}}=0\) 且 \(E_{\mathrm{exec}}\le0.75\,E_{\mathrm{id}}\) | **通过**（0.653 ≤ 0.75×1.243） |
| G_vis | \(E_q\le0.35\) 且 \(E_{\dot q}\le0.50\) | **失败**（1.018 / 1.029） |
| G_deploy | B1 \(r_{\mathrm{cat}}=0\) | 通过（不构成视觉成功） |
| G_retain | B1 \(E_{\mathrm{exec}}\le1.30\,E_{\mathrm{exec}}^{\mathrm{B0}}\) | 通过（不构成视觉成功） |

Pattern：**`perception_failure`**（预注册：¬G_vis 即此 pattern，不论 G_deploy / G_retain）。  
`structure_survives_sim_rgb_claim=false`。

## Test 主表（\(K=4\)，720 segments / 24 ep）

| 模型 | 角色 | \(E_{\mathrm{exec}}\) | \(E_{\mathrm{end}}\) | \(E_{\mathrm{id}}\) | \(r_{\mathrm{cat}}\) |
|------|------|----------------------:|----------------------:|---------------------:|---------------------:|
| B0 M2 oracle | 上限 | **0.653** | 0.746 | 1.243 | **0** |
| B1 RGB→state→M2 | 主模型 | 0.615 | 0.515 | 1.058 | 0 |
| B2 RGB latent | exploratory | 1.067 | 1.094 | 1.058 | 0 |

Perception（test）：\(E_q=1.018\)，\(E_{\dot q}=1.029\)。

探索性：\(\Delta_{\mathrm{vision}}=-0.038\)，\(\eta_{\mathrm{retain}}=1.065\)。  
**不得**据此声称视觉接口无损：G_vis 已失败，B1 的 \(E_{\mathrm{identity}}\) 用视觉 reset 而非 oracle \(x_t\)，与 B0 的 identity **不可直接混报为同一基线**。在显式 \(q,\dot q\) 未恢复时，部署误差接近 B0 只能当 **诊断异常**，不能当结构穿过视觉的证据。

B2 \(E_{\mathrm{exec}}=1.067\) 接近其 identity，视觉 latent 一步动力学 **未** 超过恒等基线。

## 读数

1. **瓶颈在感知，不在冻结 M2。** Oracle 合同下 M2 复现 X0EH 线 competent 行为。
2. **当前 \(E_{\mathrm{vis}}\)（64×64 扁平像素 + 两层 MLP）不足以从第三人称 RGB 恢复 12-DoF \(q,\dot q\)。** \(E_q\approx1\) 表示预测误差与状态 RMS 同量级。
3. **不能开 R10 / 真相机。** 仿真 RGB 接口尚未打通。
4. **禁止事后**：放宽 \(E_q/E_{\dot q}\) 阈值、改 M2、改 \(K=4\)、把 B1 的偶然 \(E_{\mathrm{exec}}\) 写成视觉成功。

## 与机制链的关系

```text
X0EH1/X0EH2  →  oracle-state capacity PASS（不改）
X0RGB        →  perception_failure
               仿真 RGB 尚未构成可用 state 接口
```

## 账本

```text
RTWX-X0RGB = RAN / perception_failure
             G_oracle=true; G_vis=false
             E_q=1.018  E_qd=1.029
             B0 E_exec^(K=4)=0.653  B1 E_exec=0.615  r_cat=0
             sim RGB ≠ real camera; R10 LOCKED
```
