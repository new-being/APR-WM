# RTWX-O0C 报告 — Composite Non-Oracle Object Pose Confirmation

日期：2026-08-29  
状态：**正式冻结** `orientation_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0C_PREREG.md`  
产物：`runs/rtwx_o0c/{run.json,metrics.json,cache_o0c_s*,run.log}`  
合同：`head+observer`；U-Net + CoordConv \(R_{CO}\)；SO(3) GeoMedian；\(p=p^{surf}-\hat R\delta_O\)；seeds **27601/02/03**；24/12/12×120  
**不改 O0G2R**；**不解锁 O1**。

## 一句话

覆盖与 localization 成立；**learned orientation 的 P90 崩到 ~91°**（med 12.8° 仍 ≤15°），故 G1 失败。  
去掉 \(R^{GT}\) 后的完整 \(T_{BO}\) **尚未**在 fresh 上闭合。

\[
\boxed{\texttt{orientation\_failure}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 | any≥0.98；seed≥0.95；joint det≥0.95 | **PASS**（1.000 / 全 seed；joint 0.953） |
| G1 | med \(e_R\)≤15°，P90≤30° | **FAIL**（med **12.8°**；P90 **90.7°**） |
| G2 | B3 \(E_p\)≤0.20，med≤5 cm | FAIL（描述；\(E_p=0.272\)，med **1.91 cm**） |
| G3 | info + ori≺mean + no collapse | FAIL（描述） |
| G4 | \(\eta\le1.5\) | 描述性 PASS（\(\eta\approx1.002\)） |

## 主表（aggregate）

| ID | \(E_p\) | med \(\|e_p\|\) | med \(e_R\) | P90 \(e_R\) |
|----|--------:|----------------:|------------:|------------:|
| B0 mean | 0.492 | 22.6 cm | **9.3°** | 90.0° |
| B1 GT-\(R\) ceiling | 0.271 | **1.38 cm** | — | — |
| B2 \(\hat R\)+GT mask | 0.272 | 1.89 cm | 12.8° | 90.7° |
| **B3 fully learned** | **0.272** | **1.91 cm** | **12.8°** | **90.7°** |

\(\Delta E_R\approx5\times10^{-4}\)（朝向误差几乎未拖坏 position med；主伤在 \(e_R\) 尾部）。

## Per-seed（描述）

| seed | \(P(V^{any})\) | \(E_p^{B3}\) | med \(e_R\) | no collapse |
|------|---------------:|-------------:|------------:|:-----------:|
| 27601 | 1.000 | 0.188 | 12.4° | ✓ |
| 27602 | 1.000 | 0.321 | 12.9° | ✓ |
| 27603 | 1.000 | 0.276 | 13.1° | ✓ |

## 读数

1. **失败模式清晰**：不是 coverage / fusion geometry，而是 **双视角 camera-frame orientation 尾部分布**（P90≈90°）。  
2. med 过门、P90 不过 → 典型 **重尾 / 对称混淆 / 偶发大错**，不是整体 bias。  
3. B0 ori med 甚至略优于 B3 → learned \(R\) **未稳定打赢 train-mean**（`ori_beats_mean=false`）。  
4. Position med 仍 ~2 cm 且 \(\eta\approx1\)，与 O0G2R 几何链一致；**\(E_p\) NRMSE 升高**反映多 seed / 缺失帧，不是本格主 STOP 原因。  
5. **O1 仍 LOCKED**。禁止放宽 P90 或改 O0R 门限救格。  
6. 机制压缩见 **O0C1**：`dual_view_both_wrong_mode`。

## Pattern

```text
pattern = orientation_failure
unlocks_o1 = false
R_BO_is_GT_nuisance = false
near_oracle_position = true   # descriptive; G1 failed
```
