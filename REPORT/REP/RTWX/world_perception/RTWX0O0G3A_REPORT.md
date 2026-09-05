# RTWX-O0G3A 报告 — Correspondence Availability Audit

日期：2026-08-30  
状态：**正式冻结** `correspondence_availability_characterized`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G3A_PREREG.md`  
产物：`runs/rtwx_o0g3a/{summary.json,run.json}`  
依赖：O0G3R test cache（seeds 29601/02/03）  
**描述性**；**不改** \(N_{\min}=50\)。

## 一句话

\[
P(V^{any})=1.0,\quad P_{\mathrm{enough}}=0.875
\]

缺口来自 **mask 面积极小的 visible 帧**：deficit 帧 median \(N_{\mathrm{union}}=8\)、median mask area **8 px**（64² 下杯子过小），而非 depth 失效。

\[
\boxed{\texttt{correspondence\_availability\_characterized}}
\]

## 聚合（test，4320 visible 帧）

| 指标 | 值 |
|------|-----|
| \(P_{\mathrm{enough}}\)（\(N_{\mathrm{union}}\ge50\)） | **0.875** |
| median \(N_{\mathrm{corr}}^{head}\) | 102 |
| median \(N_{\mathrm{corr}}^{obs}\) | **14** |
| median \(N_{\mathrm{union}}\) | 118 |
| dual-view 同时有效 | 90.6% |
| deficit 帧占比 | **12.5%**（539 帧） |

| seed | \(P_{\mathrm{enough}}\) |
|------|------------------------|
| 29601 | 0.842 |
| 29602 | 0.872 |
| 29603 | 0.912 |

## 关键曲线

### \(P(N_{\mathrm{union}}\ge N_{\min}\mid \text{mask area})\)

| mask area（union px） | \(P_{\mathrm{enough}}\) | n |
|----------------------|-------------------------|---|
| **0 – 19** | **0.0** | 432 |
| 19 – 87 | 0.69 | 341 |
| **≥ 87** | **1.0** | 其余 |

→ **阈值型合同**：mask area \(\lesssim 87\) 时几乎不可能凑满 50 个 union 点。

### \(P(N_{\mathrm{union}}\ge N_{\min}\mid N_{\mathrm{corr}}^{head})\)

| \(N_{\mathrm{corr}}^{head}\) | \(P_{\mathrm{enough}}\) |
|------------------------------|-------------------------|
| **0 – 7** | **0.0** |
| 7 – 68 | 0.66 |
| **≥ 68** | **1.0** |

→ observer 侧 median 仅 14 点；head 侧点数不足时 union 无法过线。

## 读数

1. **看得见 ≠ 足够点**：12.5% visible 帧处于“极小 mask”regime（median 8 px），与 O0G3R G0 的 \(P_{\mathrm{enough}}=0.875\) 一致。
2. **不宜直接改 \(N_{\min}\)** 救 O0G3R：这是 **64² observation + \(N_{\min}=50\)** 合同与物体成像尺度不匹配；若继续需 **新 observation/correspondence contract**（非事后降门）。
3. joint det=0.941（O0G3R）与 dual-valid=90.6% 同源：部分帧仅单视角有足够 mask。
4. 本格 **不**作科学 PASS/FAIL；为 O0G3R2 或新合同提供依据。

## Pattern

```text
pattern = correspondence_availability_characterized
deficit_frac = 0.125
deficit_median_n_union = 8
deficit_median_mask_area = 8
unlocks_o1 = false
```
