# RTWX-O0V 报告 — Robust Visual Coverage Contract

日期：2026-08-29  
状态：**正式冻结** `dual_view_coverage_supported`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0V_PREREG.md`  
产物：`runs/rtwx_o0v/{run.json,metrics.json,cache_o0v_seed*.npz,run.log}`  
合同：**无训练**；B0=`head`；B1=`head∨observer`（O0D2 预冻结）；seeds **25601/25602/25603**；24×120 / seed  
**不改 O0G2**；**无 pose claim**；允许预注册 **O0G2R**；O0C/O1 LOCKED。

## 一句话

`head_camera` alone 跨 seed **不稳**（agg vis=0.889；三 seed 均 &lt;0.95）。  
固定双视角 **`head∨observer`**：**agg \(P(V^{any})=0.996\ge0.98\)**，且每 seed \(\ge0.95\)。

\[
\boxed{\texttt{dual\_view\_coverage\_supported}}
\]

## Gates（B1）

| Gate | 条件 | 结果 |
|------|------|------|
| Aggregate | \(P(V^{any})\ge0.98\) | **PASS**（0.996） |
| Per-seed | 每 seed \(\ge0.95\) | **PASS**（1.000 / 1.000 / 0.987） |

## 主表

| seed | \(P_{\mathrm{vis}}^{head}\) | \(P_{\mathrm{vis}}^{obs}\) | \(P_{\mathrm{vis}}^{any}\) | head≥0.95 | any≥0.95 |
|------|----------------------------:|---------------------------:|---------------------------:|:---------:|:--------:|
| 25601 | 0.934 | 1.000 | **1.000** | ✗ | ✓ |
| 25602 | 0.849 | 1.000 | **1.000** | ✗ | ✓ |
| 25603 | 0.885 | 0.987 | **0.987** | ✗ | ✓ |
| **agg** | **0.889** | — | **0.996** | unstable | ✓ |

## 读数

1. **确认 O0G2 诊断**：head-only 覆盖是 episode/seed 依赖，不是一次坏运气（O0R 0.962 vs O0G2 0.902 vs 本格 0.85–0.93）。  
2. **禁止**再赌 head-only seed 回 0.95。  
3. 双视角合同来自 O0D2 screening，**非**本格事后选相机。  
4. **O0G2R 可预注册**：冻结 `head+observer` + O0G2 几何/U-Net/\(\delta_O\) 融合规则。  
5. 仍不解锁 O0C/O1（须 multi-view position confirmation）。

## Pattern

```text
B0 head-only = unstable (agg 0.889; all seeds <0.95)
B1 head∨observer = PASS
pattern = dual_view_coverage_supported
unlocks_o0g2r_prereg = true
unlocks_o1 = false
```
