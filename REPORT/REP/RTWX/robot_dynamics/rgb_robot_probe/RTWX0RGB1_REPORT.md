# RTWX-X0RGB1 报告 — Spatial-Temporal State Recovery

日期：2026-08-29  
状态：**正式冻结** `spatial_perception_failure`  
预注册：`REPORT/REG/RTWX/RTWX0RGB1_PREREG.md`  
产物：`runs/rtwx_x0rgb1/{run.json,metrics.json,cache_rgb_splits.npz,W_m2.npy}`  
任务：`put_object_cabinet`；seed **15601**；48/24/24 × 120；`front_camera` **128×128**；P2 \(L=4\)。  
下游：冻结 M2，\(P=1752\)；\(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)。  
**不做容量结论**；不声称真实相机。

## 一句话

第三人称 128×64 CNN（空间）以及 \(L=4\) 时序头 **仍不能** 恢复 12-DoF \((q,\dot q)\)。  
P2：\(E_q=0.968\)，\(E_{\dot q}=0.992\)，\(\#\{j:E_{q,j}<0.35\}=0\)。  
Oracle M2 仍 competent（\(E_{\mathrm{exec}}^{K=4}=0.610\)，\(r_{\mathrm{cat}}=0\)）。  
Pattern：**`spatial_perception_failure`**。未进入视觉→M2 部署解释。

## Gates（阈值未放宽）

| Gate | 条件 | 结果 |
|------|------|------|
| G_oracle | B0 competent | **通过**（0.610 ≤ 0.75×1.211） |
| G_vis（P2） | \(E_q\le0.35\) 且 \(E_{\dot q}\le0.50\) | **失败** |

因 \(E_q>0.35\)，不评 \(\eta_{\mathrm{retain}}\)，不跑科学意义上的 B1。

## Perception 主表（test）

| 仪器 | \(E_q\) | \(E_{\dot q}\) | \(n_{q,j}<0.35\) |
|------|--------:|--------------:|-----------------:|
| P0 flatten MLP 64², \(L=2\)（负对照） | 1.078 | 1.029 | — |
| P1 CNN 单帧 128² | 0.968 | 0.992 | **0 / 12** |
| **P2 CNN + temporal \(L=4\)（主）** | **0.968** | **0.992** | **0 / 12** |

相对 P0，P1/P2 仅把 \(E_q\) 从 1.08 降到 **0.97**，仍与状态 RMS 同量级。  
P1 与 P2 几乎相同：在 \(q\) 未恢复时，时序头对 \(\dot q\) **没有**可解释增益。

## Per-joint \(E_{q,j}\)（P2）

门限 0.35。最好的两维约 **0.80**（下标 2 与 8，左右臂对称位），其余 ≈1.00。

| \(j\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---:|---:|
| \(E_{q,j}\) | 1.00 | 0.89 | **0.81** | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | **0.80** | 1.00 | 1.00 | 1.00 |
| \(E_{\dot q,j}\) | 1.00 | 1.00 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 | 1.00 | 1.00 |

**不是**少数隐藏关节拖垮总体：12 维全部远高于 0.35。  
这是第三人称单视角 + 小 CNN **整体 observability / 仪器不足**，不是 wrist-only 问题。

## Oracle 部署（对照，非视觉成功）

B0 M2：\(E_{\mathrm{exec}}=0.610\)，\(E_{\mathrm{end}}=0.581\)，\(E_{\mathrm{id}}=1.211\)，\(r_{\mathrm{cat}}=0\)（720 segments）。

## 读数

1. **X0RGB 否掉的是 flatten MLP，本格否掉的是「128 CNN + \(L=4\)」仍不够恢复 \(q\)。** 视觉路线未关；当前仪器未打通。
2. **Dynamics 主线未坏。** 不得回头改 M2 或 \(K=4\)。
3. **下一格才考虑** keypoint / mask / 多视角 / 更强预训练编码器。不上 visual latent WM，不放宽 G_vis，不开 R10。

## 账本

```text
RTWX-X0RGB1 = RAN / spatial_perception_failure
             seed 15601; 128×128 CNN; L=4
             P2 E_q=0.968 E_qd=0.992; n_q_ok=0/12
             G_oracle=true; B0 E_exec^(K=4)=0.610
```
