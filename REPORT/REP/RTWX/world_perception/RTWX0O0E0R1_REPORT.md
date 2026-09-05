# RTWX-O0E0R1 报告 — Reference/Centering Oracle Audit

日期：2026-08-31  
状态：**正式冻结** `quotient_viable_under_oracle_cloud`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R1_PREREG.md`  
输入：O0E0P0 cache + O0E0R0 `science_cache/`（learned mask，无重训）  
产物：`runs/rtwx_o0e0r1/{summary.json,run.json,header.json}`  
seed **39601**；test **n=200**；**零训练**

## 一句话

R0 的 74.5° axis failure **不是** frozen quotient \((h,r)\) objective 在 partial geometry 下不可行；在 **GT mask + GT center** oracle ceiling 上 median **0.17°**。主因是 **controlled-pose 下 learned segmentation → 劣质 fused cloud**；reference centering 单独 oracle 化（\(B2_{pGT}\) on learned cloud）**不能**救 axis。

\[
\boxed{\texttt{quotient\_viable\_under\_oracle\_cloud}}
\]

O0E0R0 的 `effective_axis_failure` **不变**。

## 解释树（冻结）

| Branch | 结果 | 读法 |
|--------|------|------|
| **D ceil** GTmask + GT \(p\) | median **0.17°**，P90 **0.26°** | quotient geometry **可行** |
| **D seg** GTmask + reference | median **20.3°**，P90 32.2° | reference 仍有耦合，但远好于 74° |
| **D formal** learned + reference | median **74.6°** | = R0 |
| **C** \(B2_{pGT}\) learned cloud | median **74.4°** | oracle center **不**解 axis |
| **B** position ceiling + GT \(n\) | learned **9.6 cm** → GTmask **1.75 cm** | segmentation 强耦合 |
| **A** \(\|\delta^{GT}-\delta_O\|\) (learned \(\hat p^{surf}\)) | median **9.6 cm** | 与 B 同量级；**confounded** by learned mask |

**不**触发 `axis_failure_reference_coupled`（C 未过 G1 尺度）。  
**触发** `axis_failure_segmentation_coupled`（mask median 改善 **7.8 cm**）。

## \(2\times2\) Oracle Grid（B2 axis）

| Mask | Center | median \(e_{\mathrm{axis}}\) | P90 | G1 尺度 |
|------|--------|------------------------------|-----|---------|
| learned | reference | **74.6°** | 129.2° | FAIL |
| **GT** | reference | **20.3°** | 32.2° | median 近 PASS |
| learned | **GT \(p\)** | **74.4°** | 109.8° | FAIL |
| **GT** | **GT \(p\)** | **0.17°** | **0.26°** | **PASS** |

## 机制收束

\[
\boxed{
\text{R0 failure} \approx
\underbrace{\text{learned mask cloud corruption}}_{\text{dominant}}
+
\underbrace{\text{reference coupling on bad cloud}}_{\text{secondary}}
}
\]

\[
\boxed{
\text{NOT: frozen }(h,r)\text{ quotient objective insufficient under true partial geometry}
}
\]

对架构的含义：

1. **不应**在 R0 上 retune Fibonacci/search。  
2. **应**把下一格预算放在 controlled-pose **segmentation / cloud extraction**（或 GT-mask ceiling 级别的 domain 闭合），而非改 B2 objective。  
3. \(\hat p^{surf}-\delta_y n\) 在 **GT cloud** 上 position 可达 **1.75 cm**；natural upright 下有效的 fixed \(\delta_O\) 在 multi-tilt 上通过 learned mask 会表现为 view-dependent bias（A），但 GT mask 下 position 闭合 → A 的 9.6 cm 主要是 **cloud 错** 而非 \(\delta_O\) 理论失效。

## Patterns

```text
pattern = quotient_viable_under_oracle_cloud
patterns = [
  oracle_audit_complete,
  reference_offset_not_invariant,
  quotient_viable_under_oracle_cloud,
  axis_failure_segmentation_coupled
]
unlocks_reference_repair_cell = true   # segmentation/cloud upstream
unlocks_objective_hypothesis_cell = false
unlocks_o0e1 = false
unlocks_o1 = false
r0_effective_axis_failure_unchanged = true
```

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| quotient B2 **在 oracle cloud 上**闭合 axis | O0E0R0 应改判 PASS |
| controlled tilt 暴露 **learned seg** 为上游主因 | fixed \(\delta_O\) 理论被 A 单独证伪（A 用 learned surf） |
| 下一格应修 **cloud extraction**，非 B2 search | 已解锁 O0E1 / O1 |
