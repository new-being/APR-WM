# RTWX-O0G2R 报告 — Dual-View Geometry-Mediated Position Confirmation

日期：2026-08-29  
状态：**正式冻结** `dual_view_geometry_position_supported`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G2R_PREREG.md`  
产物：`runs/rtwx_o0g2r/{run.json,metrics.json,cache_*,run.log}`  
合同：`head+observer`；shared U-Net；\(\delta_O\) 资产先验；融合=逐坐标 median；\(R_{BO}^{GT}\) nuisance；seed **26601**；48/24/24×120  
**不改 O0V/O0G2**；允许预注册 **O0C**；O1 LOCKED。

## 一句话

在 O0V 双视角覆盖合同下，冻结 localization+geometry 组合在 fresh seed 上稳定恢复 \(p^{pose}\)：  
**B2 \(E_p=0.150\)，med \(1.44\) cm（strong）**，且 **B2≈B1**（\(\eta\approx1\)）。

\[
\boxed{\texttt{dual\_view\_geometry\_position\_supported}}
\]

`localization_near_oracle = true`

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 | \(P(V^{any})\ge0.98\) | **PASS**（1.000） |
| G1 | B1 \(E_p\le0.20\)，med≤5 cm | **PASS**（0.150；**1.44 cm** strong） |
| G2 | joint det \(\ge0.95\) | **PASS**（0.987；IoU_h=IoU_o=1.000） |
| G3 | B2 \(E_p\le0.20\)，med≤5 cm | **PASS**（0.150；**1.44 cm** strong） |
| G4 | \(E_p^{B2}\le0.85 E_p^{B0}\) | **PASS**（0.431×B0） |
| G5 | \(\eta\le1.5\) 或 \(\Delta E\le0.05\) | **PASS**（\(\eta\approx1.000\)） |

## 主表

| ID | \(E_p\) | med \(\|e_p\|\) | 注 |
|----|--------:|----------------:|----|
| B0 train mean | 0.348 | 24.9 cm | |
| B1 oracle dual | 0.150 | **1.44 cm** | ceiling |
| **B2 learned dual** | **0.150** | **1.44 cm** | **主** |
| B3 head-only（次） | 0.250 | 1.56 cm | 非主；frac_finite 更低 |

## 读数

1. **position interface 成立**：双视角 + 冻结 U-Net + \(\delta_O\) + median fusion 在 fully fresh 上复现可用 \(p^{pose}\)。  
2. **learned 不再是主误差源**：B2≈B1；剩余误差来自几何/融合/深度合同，而非 segmentation。  
3. **B3 描述性**：head-only \(E_p\) 更差，与 O0V coverage 结论一致；不推翻 B2。  
4. \(R_{BO}\) 仍为 GT nuisance；**完整非-oracle pose → O0C**。  
5. **可预注册 O0C**；O1 仍 LOCKED until O0C PASS。

## Pattern

```text
pattern = dual_view_geometry_position_supported
localization_near_oracle = true
unlocks_o0c_prereg = true
unlocks_o1 = false
R_BO_is_GT_nuisance = true
```
