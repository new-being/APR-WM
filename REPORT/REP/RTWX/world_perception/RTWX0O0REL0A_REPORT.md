# RTWX-O0REL0A 报告 — Ego Transform vs Visibility-Overlap Audit

日期：2026-08-31  
状态：**正式冻结** `overlap_support_hypothesis_supported`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0REL0A_PREREG.md`  
产物：`runs/rtwx_o0rel0a/`；复用 REL0 cache seed **37606**；**零训练**

## 约束

REL0 formal pattern **不变**：

\[
\boxed{\texttt{ego\_motion\_compensation\_failure}}
\]

本格仅做 evaluator-only oracle diagnostics。

## 一句话

\[
\boxed{
\text{C0 的 20° 主要来自 partial-surface overlap collapse，
而非 }T_{BC}\text{ 补偿错误或 segmentation 退化。}
}
\]

## Diagnostic A — GT-mask ICP（C0）

| Path | median | P90 |
|------|-------:|----:|
| S0 mask | 19.9° | 35.1° |
| **GT mask** | **20.0°** | **35.1°** |

GT-mask 与 S0 几乎相同 → **segmentation 排除**为主因。

## Diagnostic B — CAD 一致性（C0，静止物体）

| 量 | median | / \(D_O\) |
|----|-------:|----------:|
| \(d_0\)（\(\mathcal P_0\to T_{BO}\mathcal G\)） | **0.18 cm** | **0.020** |
| \(d_1\)（\(\mathcal P_1\to T_{BO}\mathcal G\)） | **0.18 cm** | **0.020** |
| \(\rho_{\mathrm{overlap}}\) | **0.04** | — |

\[
\boxed{
d_0,d_1\ll D_O \ \land\ \rho\approx0
\Rightarrow
\text{两帧各自在正确 world geometry 上，但看到不同 surface。}
}
\]

## Diagnostic C — \(e_{\Delta n}\) vs \(\rho_{\mathrm{overlap}}\)（pooled）

| \(\rho\) bin | \(n\) | median \(e_{\Delta n}\) | P90 |
|--------------|------:|--------------------------:|----:|
| \([0,0.1)\) | 171 | **20.0°** | 35.1° |
| \([0.1,0.25)\) | 39 | 10.5° | 35.1° |
| \([0.25,0.5)\) | 225 | **4.8°** | 16.8° |
| \([0.5,1.0]\) | 165 | **4.2°** | 13.0° |

跨 C0/C1/C2 一致：**ICP success 由 shared-surface overlap 控制，而非 camera-static/moving 二分类。**

## Pattern（descriptive）

```text
descriptive_pattern = overlap_support_hypothesis_supported
tags = [view_disjoint_surfaces_c0, segmentation_excluded_c0]
supports_overlap_gated_relative_tracking = true
```

## Beam 含义

1. relative tracking branch **升为主候选**（C1/C2 4.5°/0.66 cm vs A1 ~20°）
2. persistent surface → **low-overlap re-anchor**，非继续修 single-frame reference
3. **不调** ICP 超参；下一机制：**overlap/validity gate → adaptive routing**
