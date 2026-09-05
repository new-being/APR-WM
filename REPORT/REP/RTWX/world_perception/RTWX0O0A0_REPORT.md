# RTWX-O0A0 报告 — Appearance-Conditioned Yaw Observability

日期：2026-08-30  
状态：**正式冻结** `appearance_yaw_generalization_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0A0_PREREG.md`  
产物：`runs/rtwx_o0a0/{summary.json,run.json,console.log}`  
数据：fresh base snapshots **48/24/24**；每 base 24×15° yaw；oracle mask crop 128²；head+observer；**不** step physics  
编码器：CoordConv + spatial flatten（**无 GAP**；与 O0D3/O0R 合同）；训满 40 epoch  
O0 simulator GT 仍为 full \((T,R)\)。O0Q* 旁支仍 FROZEN。O0A1 LOCKED。  
**后续（2026-08-31）**：O0T0 预注册保留但 **NOT RUN / DEPRIORITIZED**；主线转向 O0E0 effective pose \((p,n)\)。

## 一句话

在把 \(p,n_{\mathrm{cup}},q,\mathrm{camera}\) 全部 oracle-condition 掉之后，当前 `021_cup` 的单帧 RGB 外观**不能**稳定编码杯轴 yaw：train/fresh 都停在均匀 24 类，90° catastrophe 原样还在。

\[
\boxed{\texttt{appearance\_yaw\_generalization\_failure}}
\]

这不是因果 quotient claim，只说明：

\[
\boxed{\text{single-frame RGB-D cannot close full }R}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 render | \(\Delta RGB\approx0\)；yaw 后 \(p,n\) 不变；label 与 sim \(R\) 一致 | **PASS**（repeat=0；pn/label 全过） |
| G1 B1 RGB fresh | median \(\le15^\circ\) 且 P90 \(\le30^\circ\) | **FAIL**（median=**90°**；P90=**165°**） |
| G2 tail（报） | \(f_{90}=P(e\ge75^\circ)\) | **0.625**（= 均匀 24 类） |
| G3 \(\Delta e\)（报） | \(e_{\mathrm{depth}}-e_{\mathrm{RGB}}\) | **0**（两支同为机会水平） |

n_test = 24 bases × 24 yaw = **576**。

## 主表（fresh）

| 支路 | median \(e_\theta\) | P90 | \(f_{90}\) | Top1 | Top3 | Top5 | train CE |
|------|--------------------:|----:|----------:|-----:|-----:|-----:|---------:|
| B0 depth | 90° | 165° | 0.625 | 0.042 | 0.125 | 0.208 | 3.178 |
| **B1 RGB** | **90°** | **165°** | **0.625** | **0.042** | **0.125** | **0.208** | **3.178** |
| B2 RGB-D | 90° | 165° | 0.625 | 0.042 | 0.125 | 0.208 | 3.178 |

\(\ln 24 \approx 3.178\)。Top-k 恰为 \(k/24\)。

Train 与 fresh **同为机会水平**（B1 train Top1=0.042）。不是「train 过、fresh 挂」的过拟合故事。

## 1-NN confirmatory（非门）

像素 RGB 1-NN（train→fresh）：Top1=**0.109**（机会 0.042）；median=75°；P90=165°。  
有极弱的像素残差，但远不够 15°/30° 门，也不能消掉 90° 尾。

## 读数

1. **G0 闭合**：yaw 仪器不是渲染/标签 bug。同 yaw 重复 render \(\Delta RGB=0\)；\(R\leftarrow RR_y\) 保持 \(p,n_{\mathrm{cup}}\)。
2. **几何负对照符合 O0G6A**：B0 depth 完全机会水平；换 crop 没有造出新的几何 yaw cue。
3. **Appearance 也没有**：B1 在 oracle mask、双视角、128²、无 GAP 下，40 epoch CE 不动。合成 stripe smoke 能把 CE 从 3.18 降到 2.42（仪器能学到**有**外观 cue 的物体）；真实 `021_cup` 学不到。
4. **O0C 的 ~90° 重尾在这里被钉成 yaw 本身不可观**：不是网络同时学 \(p\) 和 \(R\) 的容量问题。
5. **不是 causal quotient**：本格不问 task dynamics 是否等价。O0 target 仍是 full \((T,R)\)。

## Pattern

```text
pattern = appearance_yaw_generalization_failure
G0 = PASS
G1_rgb = FAIL
G1_depth = FAIL
f90 = 0.625
train_ce = ln(24)
rgb_pixel_nn_top1 = 0.109
unlocks_o0a1_prereg = false
unlocks_o0t0_prereg = true
unlocks_o1 = false
o0_target_unchanged = true
```
