# RTWX-O0E0R1 预注册 — Reference/Centering Oracle Audit

日期：2026-08-31  
状态：**已冻结 / RAN** `quotient_viable_under_oracle_cloud`  
依赖：O0E0R0=`effective_axis_failure`（B2 **TESTED & FAIL**）；O0E0P0 cache  
**禁止**：训练；重采；筛帧；改 B2 搜索/objective；回头改 R0 pattern；开 O0E1 / O1。

## 科学问题（唯一）

\[
\boxed{
74.5^\circ\text{ 的 axis failure，有多少来自错误 visible-surface reference，
有多少来自 quotient-axis objective 本身？}
\]

## 数据合同

| 项 | 值 |
|----|-----|
| 输入 | **同一** O0E0P0 `cache/` + O0E0R0 `science_cache/`（learned mask；**不重训** UNet） |
| GT | **仅** evaluator / oracle-substitution branch |
| B2 搜索 | 与 R0 完全冻结（Fibonacci 162；Top-8；\(5^\circ/3^\circ/1^\circ\)；2D KDTree；mean-min-sq） |

## Branches（全部 diagnostic）

### A — fixed \(\delta_O\) 假设审计

\[
\delta_{O,t}^{GT}=R_{BO,t}^{GT\top}(\hat p_t^{surf}-p_t^{GT}),\qquad
e_{\delta,t}=\|\delta_{O,t}^{GT}-\delta_O\|
\]

\(\hat p^{surf}\) 用 **learned-mask** fused cloud（与正式链一致）。  
按 \(\beta\)、yaw octant 分桶。  
**不**改 gate；解释性 pattern：`reference_offset_not_invariant` 若 tilt-bin median \(e_\delta\) spread \(>1.5\) cm。

### B — mask substitution（GT axis position ceiling）

\[
\hat p^{ceil}=\hat p^{surf}-\delta_y n^{GT}
\]

比较 learned-mask vs GT-mask 的 \(e_p\)。

### C — B2 oracle center \(B2_{pGT}\)

\[
d_i=x_i^B-p^{GT}
\]

（非 \(\hat p(n)\)）。其余 B2 冻结。

### D — \(2\times2\) oracle cloud ceiling

| ID | Mask | Center | 轴估计 |
|----|------|--------|--------|
| formal | learned | \(\hat p^{surf}-\delta_y n\) | R0 已知 |
| seg | **GT** | reference | 隔离 segmentation |
| ctr | learned | **\(p^{GT}\)** | 隔离 centering |
| ceil | **GT** | **\(p^{GT}\)** | partial geometry B2 ceiling |

轴 diagnostic 阈值（**只读**，非解锁门）：median \(\le15^\circ\)，P90 \(\le30^\circ\)（与 G1 同尺度）。

## Patterns（冻结解释树）

| Pattern | 条件 |
|--------|------|
| `reference_offset_not_invariant` | A：tilt-bin median \(e_\delta\) max−min \(>1.5\) cm |
| `quotient_viable_under_oracle_cloud` | D ceil：median \(\le15^\circ\) 且 P90 \(\le30^\circ\) |
| `quotient_insufficient_under_oracle` | D ceil 不过上式 |
| `axis_failure_reference_coupled` | formal B2 不过 G1 尺度，且 C \(B2_{pGT}\)（learned mask）过 |
| `axis_failure_segmentation_coupled` | B：GT-mask \(e_p\) 比 learned-mask 改善 \(>3\) cm median |
| `oracle_audit_complete` | 全 branch 跑完 |

**不**修改 O0E0R0 的 `effective_axis_failure`。  
**不**解锁 O0E1 / O1。

## 解释树（冻结）

1. **ceil PASS** → quotient geometry 可行；再读 seg/ctr 分支归因 upstream。  
2. **ceil FAIL** → 即使 mask+center oracle，\((h,r)\) objective 不足；授权 **新 hypothesis cell**（非 retune R0）。
