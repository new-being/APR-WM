# RTWX-O0G6A 报告 — Whole-Object Geometry Orientation Observability

日期：2026-08-30  
状态：**正式冻结** `global_geometry_ambiguous`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G6A_PREREG.md`  
产物：`runs/rtwx_o0g6a/{summary.json,run.json}`  
数据：O0G5C 128² cache（33601/02/03）；GT mask；\(p^{GT}\) nuisance；K=72 相对 \(R^{GT}\)；单向 obs→CAD；**无 ICP**  
canonical-identity 分支 **CLOSED**。O0G5R / O0G6R / O0C2 / O1 LOCKED。

## 一句话

可见整体形状对 **倾倒** 有极强约束，对 **绕杯轴的 yaw** 几乎无约束。  
full \(SO(3)\) 不能只靠当前 partial CAD 几何恢复——这和 O0C 的 90° 重尾、identity 学不出，是同一件几何事实。

\[
\boxed{\texttt{global\_geometry\_ambiguous}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-support | \(P(N\ge64)\ge0.90\)；seed≥0.85；median \(\sqrt{S(I)}/D_O\le0.15\) | **PASS**（0.915；RMS\(_I\)=**0.0205** \(D_O\)） |
| G1-top1 | \(P(d(R^*,R^{GT})\le15^\circ)\ge0.80\)；seed≥0.70 | **FAIL**（**0.478**；seeds 0.547 / 0.421 / 0.470） |
| G2-margin | median \(m>0\) | **FAIL**（\(m\approx0\)） |

n=3954 帧。GT Chamfer 贴合，不是 CAD 坐标系 bug。

## 主表

| 量 | 值 |
|----|-----|
| \(P_{\mathrm{top1}}\) | **0.478** |
| Top-5 basin recall | **1.000** |
| median \(d(R^*,R^{GT})\) | **90°** |
| P90 \(d^*\) | 180° |
| \(P(e_{R^*}\in[75,105])\) | 0.256 |

## 90° competitor（本格核心诊断）

| 模式 | median \(S/S(I)\) | \(P(S(I)<S_{\mathrm{mode}})\) |
|------|------------------:|------------------------------:|
| **Ry90**（杯轴） | **1.016** | 0.627 |
| **Ry180** | **0.998** | **0.489** |
| Ry270 | 1.014 | 0.615 |
| Rx90 / Rz90（倾倒） | **~462** | **1.000** |
| Rx180 / Rz180 | ~1003 | 1.000 |

\[
S(R^{GT})\approx S(R^{GT}R_{Y,90/180}),
\qquad
S(R^{GT})\ll S(R^{GT}R_{X/Z,90}).
\]

## 读数

1. **不是 “几何没有 orientation signal”**：把杯子推倒（绕 X/Z 转 90°）的 cost 高两个数量级。轴方向/倾角是可观测的。
2. **full \(R\) 的不可观测量是绕对称轴的 yaw**（无把杯子 ≈ 旋转体）。Ry180 甚至经常略便宜——Top-1 被 yaw 模抢走，median \(d^*=90^\circ\)。
3. **Top-5=1 不能开 O0G6R**：basin 总在短名单里，是因为 yaw 模与 GT **同 cost**，不是 “搜一下就能唯一化”。ICP 不能从退化 landscape 里造出偏航。
4. **与前序闭合**：O0C 重尾 90°、anchor identity fresh FAIL，都与 “局部/全局几何不编码杯轴方位” 一致。不是网络容量问题。
5. **刚体最小状态应改写**：需要的不是 pixel identity，而是 **task-relevant quotient**（例如轴方向 + 位置；或 appearance 才能打破 yaw）。

## Pattern

```text
pattern = global_geometry_ambiguous
G0_support = PASS
G1_top1 = FAIL
G2_margin = FAIL
tilt (Rx/Rz) = uniquely excluded
yaw about cup axis (Ry) = degenerate
unlocks_o0g6r_prereg = false
unlocks_o1 = false
```
