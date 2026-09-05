# RTWX-O0G3B 报告 — Correspondence Instrument Closure

日期：2026-08-30  
状态：**正式冻结** `pixel_coord_not_representable`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G3B_PREREG.md`  
产物：`runs/rtwx_o0g3b/{summary.json,run.json}`  
依赖：O0G3R=`coverage_failure`；复用 `runs/rtwx_o0g3r` train cache  
**无** fresh 科学 claim；**不**解锁 O0G3R2 / O0C2 / O1。

## 一句话

**Target pipeline 闭合**（lookup 记住训练集）；但 \((u,v,d)\to x^O\) 与 crop-RGBD 均无法在 train 上压到 0.05。  
O0G3R 的 correspondence 失败**不是** normalization/indexing bug，而是 **从当前输入到 dense \(x^O\) 的表示路径未闭合**。

\[
\boxed{\texttt{pixel\_coord\_not\_representable}}
\]

## 仪器支路

| 支路 | train \(E_{\mathrm{corr}}^{\mathrm{norm}}\) | 门 0.05 | 解读 |
|------|---------------------------------------------|---------|------|
| **B0** index lookup | **0.0027** | **PASS** | GT target / \(D_O\) / evaluator 无 pipeline bug |
| **B1** \((u,v,d)\to x^O\) MLP | 0.300 | FAIL | 跨 episode 池化时 \((u,v)\) 与 canonical 点**不对齐**（物体运动） |
| **B2** oracle-mask crop RGBD | 0.479 | FAIL | 与 O0G3R whole-frame U-Net 一致：dense \(x^O\) 难记 |
| Audit \(\phi_i\approx\phi_j\) | \(f_{\mathrm{amb}}=0.0\) | — | 当前 \(\tau\) 下未触发 ambiguity pattern |

\(n_{\mathrm{train\_pixels}}=16384\)（3-seed train 子采样）。

## 读数

1. **B0 PASS** 排除 `target_pipeline_failure`：O0G3R 的 \(E_{\mathrm{corr}}\approx0.52\) **不是** target 构造或评估器错误。
2. **B1 FAIL** 需在解读时注明：实现将**多 episode** 像素池化到同一 \((u,v,d)\) 空间；固定相机下物体运动使同一 \((u,v,d)\) 对应不同 \(x^O\)——该诊断支路对 moving object **不适定**，不能单独证明“MLP 容量不足”，但说明 **无 pose/frame 上下文的局部坐标不足以承载 dense correspondence**。
3. **B2 FAIL** 与 O0G3R RGBD U-Net 一致：即使 oracle mask crop，仍无法 memorization → 问题在 **dense canonical 目标本身**，非仅 whole-frame sparsity。
4. Audit 未达 10% ambiguity 门；**不能**据此排除纹理弱导致的局部歧义，但当前 audit 未把它标为主因。
5. **不解锁** O0G3R2；应考虑假设切换：**sparse axes / keypoints** 或 **CAD descriptor matching**（见路线讨论）。

## Pattern

```text
pattern = pixel_coord_not_representable
B0_lookup = PASS
B1_uvd_mlp = FAIL
B2_crop_rgbd = FAIL
unlocks_o0g3r2_prereg = false
unlocks_o1 = false
```
