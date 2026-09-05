# RTWX-O0C1 报告 — Orientation Tail Mechanism Audit

日期：2026-08-29  
状态：**诊断冻结** `dual_view_both_wrong_mode`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0C1_PREREG.md`  
产物：`runs/rtwx_o0c1/{run.json,metrics.json,run.log}`  
合同：复用 O0C caches（27601/02/03）；同一 CoordConv \(R_{CO}\) + GeoMedian；**不改 O0C**；**不解锁 O1**。

## 一句话

尾部不是 fusion 把好结果融坏，也不是 \(C_4\)-Y 对称误标；**catastrophe 时两个视角通常同时掉进错误 mode**（约 86%），且错误质量集中在 ~90°。

\[
\boxed{\texttt{dual\_view\_both\_wrong\_mode}}
\]

## D1 — 分布

| 区间 | fusion 计数 | 占比 |
|------|------------:|-----:|
| [0,15) | 2189 | 0.507 |
| [15,30) | 1226 | 0.284 |
| [30,60) | 306 | 0.071 |
| [60,120) | 505 | 0.117 |
| [120,180] | 94 | 0.022 |

- fusion：med **14.9°**，P90 **90.0°**  
- \(\mathrm{frac}(e_R\in[75,105])\approx0.096\)；near-180 ≈0.003  
→ **离散 ~90° mode**，不是连续重尾或 180° 翻转。

## D2 — 分视角 / catastrophe 四分类（\(e_f>30^\circ\)，n=905）

| 类 | 计数 | 占比 |
|----|-----:|-----:|
| h✓ o✗ | 2 | 0.002 |
| h✗ o✓ | 121 | 0.134 |
| **h✗ o✗** | **782** | **0.864** |
| h✓ o✓ fusion✗ | 0 | 0.000 |

| 源 | med \(e_R\) | P90 |
|----|------------:|----:|
| head | 24.5° | 62.2° |
| observer | 9.1° | 85.1° |
| fusion | 14.9° | 90.0° |

## Oracle \(e_{\mathrm{best}}=\min(e_h,e_o)\)

| | med | P90 |
|--|----:|----:|
| best | 8.8° | **82.0°** |
| fusion | 14.9° | 90.0° |

`implies_fusion_selection_issue = false`（best 的 P90 仍然 catastrophic）。

## D3 — cross-view disagreement

| \(d_{\mathrm{view}}\) | n | \(P(e_f>30^\circ)\) |
|----------------------|--:|-------------------:|
| [0,15) | 941 | 0.078 |
| [15,30) | 1752 | 0.029 |
| [30,60) | 918 | 0.313 |
| [60,90) | 178 | 0.629 |
| [90,180] | 22 | 0.955 |

→ disagreement 是可用的 **uncertainty 信号**（本格不扫 τ 救格）。

## D4 — symmetry

- \(G_{\mathrm{sym}}=C_4\) about object Y：catastrophe 上 med \(e_R^{\mathrm{sym}}=67.7^\circ\)，**resolved 仅 0.090**  
- continuous yaw audit 在 cata 上 med 86.5° / P90 139°  
→ **排除** `orientation_state_symmetry_mismatch`（至少在该 \(G_{\mathrm{sym}}\) 下）。

## D5 — 条件（次要）

- 单视角占比：cata **0.399** vs ok **0.043**  
- head mask 面积：cata 49 vs ok 98  
→ 有 view/面积相关，但主类仍是双错 mode，不改写 pattern。

## 读数

1. **根因**：orientation representation / generalization 的 **双视角共模错误**，质量 ~90°。  
2. **不是**：fusion 破坏（fusion✗=0）；不是 oracle view-select 可救（best P90 仍 82°）；不是本格 \(C_4\)-Y 对称商。  
3. Position 线无需再碰。  
4. 下一候选应针对 **geometry-mediated \(R\)** 或更强 correspondence，而不是加大 CoordConv / 改 GeoMedian。  
5. \(d_{\mathrm{view}}\) 可留作后续 uncertainty / active sensing 输入。  
6. **O1 LOCKED**。

## Pattern

```text
pattern = dual_view_both_wrong_mode
diagnostic_only = true
unlocks_o1 = false
does_not_rewrite_o0c = true
```
