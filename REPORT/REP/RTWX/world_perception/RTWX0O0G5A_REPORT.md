# RTWX-O0G5A 报告 — CAD Descriptor Observability（冻结 FPFH）

日期：2026-08-30  
状态：**正式冻结** `local_descriptor_ambiguous`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G5A_PREREG.md`  
产物：`runs/rtwx_o0g5a/{summary.json,run.json,cache_o0g5a_s*}`  
合同：128² RGB-D；head+observer；Open3D FPFH；CAD `021_cup/visual/base0.glb` 8192 点；seeds **31601/02/03**；12×120；**无训练、无 RANSAC**  
**不**解锁 O0G5R / O0C2 / O1。

## 一句话

128² 下表面采样 **够**（\(P_{\mathrm{support}}=0.973\)）。冻结 FPFH **不能**把可见点检索到正确 CAD 区域：med \(e_{\mathrm{match}}^{norm}\approx8.0\)（约 \(8D_O\)），Top-5 recall \(\approx0\)。  
纯局部 CAD geometry 不足以确定 canonical identity。

\[
\boxed{\texttt{local\_descriptor\_ambiguous}}
\]

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0-support | \(P(N_{\mathrm{usable}}\ge64)\ge0.90\)；每 seed ≥0.85 | **PASS**（0.973） |
| G1-ambiguity | \(f_{\mathrm{amb}}<0.15\) | **PASS**（0.0057） |
| G2-matching | med ≤0.10；P90 ≤0.25；Top-5 ≥0.75 | **FAIL**（med **8.00**；P90 **11.55**；Top-5 **0.000**） |

按预注册：G0 ∧ G1 ∧ ¬G2 → `local_descriptor_ambiguous`。

G1 的 pairwise 近邻歧义门过了，但 **检索本身完全不对**：正确 CAD 点几乎不在 Top-5。这不是“近邻互撞”，而是 **FPFH 与 object-frame identity 无对齐**。

## 主表

| seed | \(P_{\mathrm{support}}\) | med \(e^{norm}\) | P90 | Top-5 | \(f_{\mathrm{amb}}\) |
|------|-------------------------:|-----------------:|----:|------:|----------------------:|
| 31601 | 0.924 | 7.93 | 11.49 | 0.000 | 0.0069 |
| 31602 | 0.997 | 8.04 | 11.57 | 0.000 | 0.0049 |
| 31603 | 0.997 | 8.04 | 11.56 | 0.000 | 0.0056 |
| **agg** | **0.973** | **8.00** | **11.55** | **0.000** | 0.0057 |

## 读数

1. **Observation contract 已换档**：相对 O0G3A 的 64² / \(N_{\min}=50\) 失败，128² + \(N_{\mathrm{desc}}=64\) **过 G0**。表面信息足够，不是分辨率阻塞。
2. **Frozen FPFH 无匹配信号**：median error ~8 个物体直径；Top-5 命中率实质为零。classical local 3D descriptor 不能替代 dense \(x^O\) 或固定 landmark。
3. **G1 低歧义 ≠ 可辨识**：descriptor 近邻彼此不远、但检索到的 CAD 点与 GT \(x^O\) 无关。杯子旋转对称 / 弱几何使 FPFH 无法锚定 global frame。
4. **不解锁 O0G5R**（descriptor matching + robust registration）。下一格应按预注册进入 **learned / global-context CAD descriptors**，不是再扫 FPFH 半径。
5. **O0C2 / O1 LOCKED**。

## Pattern

```text
pattern = local_descriptor_ambiguous
G0_support = PASS
G1_pairwise_ambiguity = PASS (not the failure mode)
G2_matching = FAIL (med=8.0 D_O, top5≈0)
unlocks_o0g5r_prereg = false
unlocks_o1 = false
```
