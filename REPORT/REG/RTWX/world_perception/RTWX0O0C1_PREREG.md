# RTWX-O0C1 预注册 — Orientation Tail Mechanism Audit

日期：2026-08-29  
状态：**已冻结（诊断格）**  
依赖：O0C=`orientation_failure`；复用 O0C caches / 同一 orientation 合同  
**禁止**：改 O0C 门限/pattern；放宽 P90；解锁 O1；arch sweep；把本格写成 scientific PASS。

## 科学问题

> \(P_{90}(e_R)\approx90^\circ\) 的尾部到底从哪里来？

不训练新系统；不声称完整 \(T_{BO}\) 恢复。只做机制归因。

## 数据合同

| 项 | 值 |
|----|------|
| 源 | O0C frozen caches（seeds **27601/02/03** test；train 仅用于复现同一 Ori 模型） |
| 网络 | 与 O0C 相同 CoordConv \(RGB\to R_{CO}\)，共享；fusion=SO(3) chordal GeoMedian |
| 禁止 | 换相机；改 fusion 权重；test 后选模型 |

## 诊断块（冻结）

| ID | 内容 |
|----|------|
| D1 | \(e_R\) 分箱：`[0,15),[15,30),[30,60),[60,120),[120,180]`；是否集中 ~90° / ~180° |
| D2 | 分视角 \(e_h,e_o,e_f\)；catastrophe（\(e_f>30^\circ\)）四分类：`(h✓o✗)/(h✗o✓)/(h✗o✗)/(h✓o✓ fusion✗)` |
| D3 | \(d_{\mathrm{view}}=d_{SO(3)}(\hat R_h,\hat R_o)\) vs \(P(e_f>30^\circ\mid d_{\mathrm{view}})\)（**不扫 τ 救格**） |
| D4 | \(G_{\mathrm{sym}}=C_4\) about object **Y**（asset 近似竖直轴；另报 continuous yaw audit）；\(e_R^{\mathrm{sym}}=\min_g d(\hat R,Rg)\) |
| D5 | catastrophe 与 mask area / 单视角可见 / seed 的关联 |
| Oracle | \(e_{\mathrm{best}}=\min(e_h,e_o)\)：若 \(P_{90}(e_{\mathrm{best}})<30^\circ\) 而 fusion P90 大 → fusion/selection |

## Diagnostic patterns（非 scientific PASS）

| Pattern | 证据优先 |
|---------|----------|
| `orientation_state_symmetry_mismatch` | catastrophe 在 \(e_R^{\mathrm{sym}}\) 下多数消失（sym med&lt;15° 且 frac_resolved≥0.7） |
| `orientation_fusion_failure` | \(P_{90}(e_{\mathrm{best}})<30^\circ\) 且 fusion P90≥30°，或 `(h✓o✓ fusion✗)` 显著 |
| `dual_view_both_wrong_mode` | \(P_{90}(e_{\mathrm{best}})\ge30^\circ\) 且 `(h✗o✗)` 主导 catastrophe |
| `view_dependent_observability` | catastrophe 与单视角/掩膜面积强相关，且非上三者主因 |
| `orientation_tail_inconclusive` | 证据冲突或不足 |

## 解锁

**不解锁 O1。** 仅允许根据 pattern **新预注册**后续格（symmetry quotient / view selection / geometry-mediated \(R\)）。  
**不回写** O0C。
