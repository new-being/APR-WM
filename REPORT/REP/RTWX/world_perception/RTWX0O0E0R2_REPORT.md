# RTWX-O0E0R1 报告 — Controlled-Pose Segmentation Repair + Contingent Science

日期：2026-08-31  
状态：**正式冻结** `segmentation_cloud_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R2_PREREG.md`  
产物：`runs/rtwx_o0e0r2/`；fresh test seed **37602**；n=**200**  
\(S_0\)=natural O0E0 cache；\(S_1\)=P0 controlled train/val；**正式 science 未运行**

## 一句话

在 fresh controlled test 上，**\(S_1\)（P0 多姿态重训）未通过 instrument**，pattern 为 `segmentation_cloud_failure`。与假设 \(H_{\mathrm{seg}}\) **相反**：**\(S_0\)（natural upright 训练）** 在 fresh controlled 上 instrument **全过**（G_I1 **0.17°**，G_I2 **1.77 cm**），而 \(S_1\) 的 G_I1 仍为 **59.3°**；主因是 **observer 视图 IoU 崩溃**（0.46），不是 B2。

\[
\boxed{\texttt{segmentation\_cloud\_failure}+\texttt{segmentation\_training\_insufficient}}
\]

## Instrument（fresh test，\(S_1\) 为判定支路）

| 支路 | \(G_{I1}\) median / P90 | \(G_{I2}\) median \(e_p\) | instrument |
|------|-------------------------|---------------------------|------------|
| **\(S_0\) natural** | **0.17° / 0.23°** | **1.77 cm** | **PASS** |
| **\(S_1\) controlled** | 59.3° / 86.2° | 1.85 cm | **FAIL**（G_I1） |

\(S_1\) G_I2 position 已过（1.85 cm），但 **G_I1 axis-cloud 未过** → STOP science。

## Mask / cloud diagnostics

| 量 | \(S_0\) | \(S_1\) |
|----|---------|---------|
| mean IoU head | **1.000** | 0.952 |
| mean IoU **observer** | **1.000** | **0.456** |
| cloud NN dist | 2.2 mm | 49 mm |
| cloud centroid disp | 2.4 mm | 50 mm |

\(S_1\)：**head 尚可、observer 严重退化** → dual-view fusion 被单侧坏 cloud 拉垮。  
各 \(\beta\) tilt 上 \(S_0\) IoU 均为 1.0；\(S_1\) observer IoU 在各 tilt 约 **0.42–0.56**（无单调 tilt 恶化，而是 observer 全局失效）。

## 与 R0/R1 的对照

| 设置 | learned mask + GT \(p\) axis median |
|------|-------------------------------------|
| R1（R0=P0-trained，P0 test） | 74.4° |
| R2 \(S_1\)（P0-trained，**fresh** test） | 59.3° |
| R2 \(S_0\)（natural-trained，fresh test） | **0.17°** |

R0/R1 的 cloud corruption **与 P0-trained UNet 的 observer 退化一致**；但 **natural \(S_0\) 在 fresh controlled 上并未失败** → \(H_{\mathrm{seg}}\)「只需把训练分布改成 controlled」**未被支持**，且方向相反。

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| \(S_1\) instrument FAIL；science STOP | O0E0R0/R1 pattern 已改 |
| \(S_0\) 在 fresh controlled 上 cloud closure 已达 oracle 尺度 | natural seg 已闭合正式 B2 claim（未跑 formal science） |
| \(S_1\) observer IoU≈0.46 是机制主因 | 应 retune B2 |
| controlled 重训 **不够/有害**（相对 \(S_0\)） | fixed \(\delta_O\) 理论被本格证伪 |

## Pattern

```text
pattern = segmentation_cloud_failure
tags = [segmentation_training_insufficient]
science_ran = false
unlocks_o0e1 = false
```

## 下一步含义

1. **B2 继续冻结**（R1 ceil 0.17° 仍成立）。  
2. **不应**再假设「P0 controlled 重训 UNet」即修复路径；需查 **为何 P0 监督训练损害 observer**（数据/视图平衡/过拟合 head），或 **直接用 \(S_0\) 自然训练链** 在 fresh controlled 上跑 contingent formal science（需新 prereg，本格未授权）。  
3. reference/centering 仍 secondary（\(S_1\) G_I2 已过）。
