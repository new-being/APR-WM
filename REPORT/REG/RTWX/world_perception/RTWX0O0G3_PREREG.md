# RTWX-O0G3 预注册 — Geometry-Mediated Orientation（G0：Oracle Correspondence Ceiling）

日期：2026-08-29  
状态：**已冻结（本文件只开 G0）**  
依赖：O0C=`orientation_failure`；O0C1=`dual_view_both_wrong_mode`；O0G2R position geometry PASS  
**禁止**：改 O0C/O0C1；direct `RGB→R`；ICP 作主法；arch sweep；解锁 O1；未 G0 PASS 不开 learned correspondence。

## 科学问题（G0 only）

> 在 partial RGB-D 双视角下，**已知 dense correspondence** 时，cup 的 \(R_{BO}\) 能否由显式刚体几何（Kabsch）稳定恢复？

\[
(x_i^{O,\mathrm{GT}},x_i^B)
\xrightarrow{\mathrm{Kabsch}}
\hat R_{BO}
\]

**不**训练网络。Learned canonical correspondence = **O0G3R**（仅 G0 PASS 后新预注册）。

## 假设对照

| 失败 | 新假设 |
|------|--------|
| \(RGB\to R\)（O0C / O0C1：~90° 共模） | \(RGB(D)\to\) correspondence \(\to\) Kabsch \(\to R\) |

## 冻结合同（G0）

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | **`head` + `observer`** |
| \(x_i^B\) | Position + `model_matrix`（OpenGL→world；与 O0G2 同线） |
| \(x_i^{O,\mathrm{GT}}\) | oracle：\(x^O=R_{BO}^{\mathrm{GT}\top}(x^B-p_{BO}^{\mathrm{GT}})\)（pose-frame 表面坐标；**非** AABB-NOCS 归一化） |
| 融合 | 双视角有效像素对 **并集** 后一次 Kabsch |
| 禁止 | ICP 主路径；learned \(R\)；改 \(\delta_O\) 用途 |
| seeds | **28601 / 28602 / 28603** |
| 规模 | 每 seed **24 × 120**；**无训练** |

## Gates（沿用 O0C / O0R orientation）

| Gate | 条件 |
|------|------|
| G0-cov | agg \(P(V^{any})\ge0.98\)；每 seed \(\ge0.95\)；每帧并集对应点数 \(\ge N_{\min}\)（冻结 \(N_{\min}=50\)） |
| G0-ori | \(\mathrm{median}\,e_R\le15^\circ\)，\(P_{90}\le30^\circ\) |

必报：

\[
P(e_R\in[75^\circ,105^\circ])
\]

（O0C1 baseline ≈0.096；几何若消 mode，应显著下降。）

Secondary（不扫 τ、不救格）：per-view Kabsch、\(d_{\mathrm{view}}\)、registration residual。

## Patterns（G0）

| Pattern | 条件 |
|---------|------|
| `orientation_geometry_insufficient` | ¬(G0-cov ∧ G0-ori) |
| `oracle_orientation_geometry_supported` | G0-cov ∧ G0-ori |

## 解锁

G0 PASS → 允许预注册 **O0G3R**（learned \(\hat x_O\) + Kabsch）。  
G0 FAIL → **STOP**；禁止直接训 correspondence 网络。  
**O1 LOCKED** until 完整非-oracle \(T_{BO}\) PASS。
