# RTWX-O0G5B 预注册 — Global-Context CAD Descriptor Instrument

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G5A=`local_descriptor_ambiguous`  
**禁止**：RANSAC/Kabsch 作主结论；连续 \(x^O\) 回归；fresh generalization；arch sweep；解锁 O0G5C/O0G5R/O0C2/O1。

## 科学问题（instrument / train memorization）

> 在 **object-level context** 下，可见点能否被赋予稳定的 **离散 CAD-anchor identity**（而非连续 \(x^O\)）？

\[
\phi_i=f_\theta(x_i,\mathcal P_{\mathrm{object}},RGBD_{\mathrm{object}}),\quad
y_i=\arg\min_j\|x_i^{O,GT}-a_j^O\|
\]

InfoNCE 匹配固定 FPS anchors。GT pose **仅用于训练标签与评价**。

## 冻结合同

| 项 | 值 |
|----|------|
| 数据 | 复用 O0G5A **128²** cache（seeds 31601/02/03；instrument 非 fresh） |
| CAD | `visual/base0.glb` × `model_data0.json` **scale**（object-frame meters） |
| Anchors | **K=512** FPS，运行前冻结（seed 32601） |
| B0 | index lookup → K-way label（pipeline） |
| B1 | local \((u,v,d,rgb)\) → \(z_i\)（learned local analogue of FPFH） |
| B2 | local + PointNet global on **centroid-centered** object cloud |
| Loss | InfoNCE；\(\tau=0.07\)；CAD 侧 learnable \(z_j^{CAD}\) |
| 子采样 | 8192 点；lookup 4096 |

B1/B2 **同一** anchor 监督。无 rotation loss。

## Gates（沿用 O0G5A matching 语义）

| 支路 | PASS |
|------|------|
| B0 | label acc \(\ge0.99\) |
| B2（主） | train med \(e_{\mathrm{match}}^{norm}\le0.10\) **且** Top-5 recall \(\ge0.75\) @ \(0.05D_O\) |

B1 为消融，不进 primary pattern。

\[
e_{\mathrm{match}}^{norm}=\|\hat a_i^O-x_i^{O,GT}\|/D_O
\]

## Patterns（互斥）

| Pattern | 条件 |
|---------|------|
| `target_pipeline_failure` | ¬B0 |
| `global_canonical_identity_not_representable` | B0 ∧ ¬B2 |
| `global_canonical_identity_supported` | B0 ∧ B2 |

仅 B2 PASS 允许预注册 **O0G5C**（fresh matching，仍无 RANSAC）。  
O0G5R / O0C2 / O1 LOCKED。
