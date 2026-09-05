# RTWX-O0G3 报告 — Geometry-Mediated Orientation（G0 Oracle Ceiling）

日期：2026-08-29  
状态：**正式冻结** `oracle_orientation_geometry_supported`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G3_PREREG.md`  
产物：`runs/rtwx_o0g3/{run.json,metrics.json,cache_o0g3_s*,run.log}`  
合同：`head+observer`；oracle \(x^O=R^{\top}(x^B-p)\) + Kabsch；seeds **28601/02/03**；24×120；**无训练**；非 ICP  
**不改 O0C**；允许预注册 **O0G3R**；O1 LOCKED。

## 一句话

在双视角 partial RGB-D 下，**已知 dense correspondence** 时 Kabsch 以近机器精度恢复 \(R_{BO}\)：med/P90 \(e_R=0\)，\(P([75,105])=0\)。  
~90° 共模错误**不是**“有对应时几何不可辨识 / Kabsch 病态”造成的。

\[
\boxed{\texttt{oracle\_orientation\_geometry\_supported}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-cov | any≥0.98；seed≥0.95；足够对应点 | **PASS**（any≈1.000；enough≈0.969） |
| G0-ori | med≤15°，P90≤30° | **PASS**（med **0**；P90 **0**） |

## 主表

| | med \(e_R\) | P90 | \(P([75,105])\) | residual RMS |
|--|------------:|----:|----------------:|-------------:|
| fusion（主） | **0** | **0** | **0** | ~1e-16 |
| head（次） | ~0 | ~0 | 0 | — |
| observer（次） | ~0 | ~0 | 0 | — |

| seed | \(P(V^{any})\) | med \(e_R\) | P90 |
|------|---------------:|------------:|----:|
| 28601 | 1.000 | 0 | 0 |
| 28602 | 1.000 | 0 | 0 |
| 28603 | 1.000 | 0 | 0 |

## 读数

1. **几何天花板成立**：oracle correspondence → Kabsch 在 cup / dual-view 合同下可辨识 \(R\)。  
2. 对照 O0C1：direct \(RGB\to R\) 的 ~90° mode **不**来自“显式几何在有对应时仍崩”；来自 **learned correspondence / rotation regression** 一侧。  
3. 本格 \(x^O\) 由 GT pose 构造 → 近零误差是预期闭环；价值在于排除 `orientation_geometry_insufficient`。  
4. **可预注册 O0G3R**：learned \(\hat x_O\) + 同一 Kabsch。  
5. **O1 仍 LOCKED**。禁止把本格写成完整 non-oracle pose PASS。

## Pattern

```text
pattern = oracle_orientation_geometry_supported
unlocks_o0g3r_prereg = true
unlocks_o1 = false
icp_not_primary = true
```
