# RTWX-O0E0R0 报告 — Controlled-Pose Effective-Pose Science Recovery

日期：2026-08-31  
状态：**正式冻结** `effective_axis_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R0_PREREG.md`  
输入：`runs/rtwx_o0e0p0`（`controlled_pose_observation_qualified`）  
产物：`runs/rtwx_o0e0r0/{summary.json,run.json,header.json,science_cache/,per_frame_b2.json}`  
seed **38601**；UNet seed **36601**；test **n=200**；B0/B1/B2 **全部正式跑完**

## 一句话

在 observation support 与 axis excitation 均已合格的受控多姿态 cache 上，冻结 quotient-CAD B2 **首次接受 fresh non-oracle 检验**，但 **axis 与 position 均未过门**。这不是 natural-distribution 问题，也不是 constant-up shortcut；而是 **learned RGB-D → fused cloud → quotient axis** 链在受控条件下仍不能闭合 \((p,n)\)。

\[
\boxed{\texttt{effective\_axis\_failure}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 detection | \(P(\hat M\neq\varnothing)\ge0.95\) | **PASS**（**1.000**） |
| G1 axis (B2) | median \(\le15^\circ\)，P90 \(\le30^\circ\) | **FAIL**（**74.5°** / **130.3°**） |
| G2 position (B2) | \(E_p\le0.20\)，median \(\le5\) cm | **FAIL**（\(E_p=0.190\)，median **11.3** cm） |
| B0 exclusion | B0 不过 G1 | **PASS**（B0 median **20.0°**，p90 **49.2°**） |

## 三支结果（test n=200）

| 支路 | median \(e_{\mathrm{axis}}\) | P90 | \(f_{90}\) | median \(e_p\) (cm) | \(E_p\) |
|------|------------------------------|-----|------------|---------------------|---------|
| B0 constant | **20.0°** | 49.2° | 0.00 | 10.9 | 0.178 |
| B1 PCA | 88.6° | 108.8° | 0.76 | 12.4 | 0.188 |
| **B2 quotient-CAD** | **74.5°** | **130.3°** | **0.50** | **11.3** | **0.190** |

- B0 被 G1 排除（非 trivial constant-up PASS）。
- B2 **劣于** B0（median 74.5° vs 20.0°）；B1 亦崩溃。
- full-\(R\) diagnostic（非门）：B2 median **127.4°**，p90 **171.5°**。

## Mechanism diagnostics（只读）

### A — 按 tilt \(\beta\)

| \(\beta\) | median \(e_{\mathrm{axis}}\) | P90 | n |
|-----------|------------------------------|-----|---|
| 0° | 63.2° | 128.8° | 40 |
| 10° | 73.0° | 135.5° | 40 |
| 20° | 70.9° | 112.9° | 40 |
| 35° | 89.7° | 114.0° | 40 |
| 50° | 89.7° | 147.8° | 40 |

各 tilt 均 FAIL；大倾角（35°/50°）略差，但小 tilt 亦远未过门 → **非单纯小倾角退化**。

### B — yaw octant（\(\beta=20^\circ\)）

8 octant 均有样本；median 约 **61°–83°**，无强周期 yaw 模式 → **不像 \(S(n)\) yaw-marginalization 机制性失效**；更像整体 cloud/registration 失败。

### C — oracle-axis position ceiling

| 量 | median \(e_p\) |
|----|----------------|
| \(\hat p^{\mathrm{ceil}}=\hat p^{\mathrm{surf}}-\delta_y n^{GT}\) | **9.9 cm** |
| B2 \(\hat p^{\mathrm{surf}}-\delta_y\hat n^{B2}\) | **11.3 cm** |

ceiling 与 B2 **均未过** 5 cm 门 → 问题不只在 axis 传播；**surface localization / reference chain** 在受控数据上亦未闭合。

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| 受控 qualified cache 上 B2 **已测**且 **FAIL** | natural task 不需要估 \(n\)（那是另一 diagnostic） |
| observation chain（det）闭合 | \(RGBD\to(p,n)\) 在受控条件下已支持 |
| B0 非 trivial shortcut（被 G1 排除） | 应改 B2 搜索再跑同一格 |
| 失败定位：whole-shape quotient + surface position 均未过 | yaw 是主因（\(f_{90}\) 高但 yaw octant 无周期） |

## Pattern

```text
pattern = effective_axis_failure
G0_P_det = 1.000
G1_B2_median_deg = 74.53
G1_B2_p90_deg = 130.29
G2_B2_median_ep_cm = 11.30
B0_excluded = true
B0_median_deg = 20.00
B2_first_tested = true
claim_scope = controlled_021_cup_rgbd_only
unlocks_o0e1_prereg = false
unlocks_o1 = false
```

## 机制收束

1. **O0E0 natural**：distribution insufficient → B2 UNTESTED。  
2. **O0E0P0**：distribution qualified → 合法 science 台。  
3. **O0E0R0**：B2 **TESTED & FAIL**；constant baseline 被排除，但 quotient-CAD 未从观测恢复 axis。

\[
\boxed{
\text{representation capability（受控）: 当前冻结 pipeline 不能估 }(p,n)
}
\]

与 natural \(r_{\mathrm{excite}}=0.126\) 的 task-necessity diagnostic **并存**，不互相抵消。

解锁：**O0E1 / O1 仍 LOCKED**。本格结果**不授权**在同一格内 retune B2 搜索或改 objective。
