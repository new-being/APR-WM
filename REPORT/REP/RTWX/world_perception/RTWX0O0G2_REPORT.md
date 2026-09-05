# RTWX-O0G2 报告 — Learned Localization + Geometry-Mediated Position

日期：2026-08-29  
状态：**正式冻结** `coverage_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G2_PREREG.md`  
产物：`runs/rtwx_o0g2/{run.json,metrics.json,cache_o0g2_*.npz,run.log}`  
合同：`head_camera` 64²；small U-Net；\(\delta_O\)=O0G1b 资产先验；seed **24601**；48/24/24×120  
**无** `RGB\to p`；**不改** O0G/O0G1b/O0R 门限；**不解锁** O0C/O1。

## 一句话

G0 coverage **未过**（\(P_{\mathrm{FOV}}=P_{\mathrm{visible}}=0.902<0.95\)）→ pattern：

\[
\boxed{\texttt{coverage\_failure}}
\]

但在同一 test 上，**oracle / learned 几何链与信息增益均通过**（B2≈B1，median \(1.62\,\mathrm{cm}\)，IoU≈1）。失败点是 **fresh 覆盖合同**，不是 localization 或 known geometry。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 coverage | FOV≥0.95，vis≥0.90 | **FAIL**（0.902 / 0.902） |
| G1 oracle | \(E_p\le0.20\)，med≤5 cm | **PASS**（0.049 / **1.62 cm**；strong） |
| G2 loc | med IoU≥0.50；empty\|vis≤0.05 | **PASS**（IoU=1.000；empty=0.001） |
| G3 pose | B2 \(E_p\le0.20\)，med≤5 cm | **PASS**（0.049 / **1.62 cm**；strong） |
| G4 info | \(E_p^{B2}\le0.85 E_p^{B0}\) | **PASS**（**0.050×** B0） |

## 主表（test）

| 仪器 | \(E_p\) | median \(\|e\|\) | 备注 |
|------|--------:|-----------------:|------|
| B0 train-mean | 0.986 | 25.2 cm | |
| B1 oracle mask+geom | 0.049 | **1.62 cm** | strong |
| **B2 learned mask+geom** | **0.049** | **1.62 cm** | \(\approx\) B1 |

分解：\(\Delta E_{\mathrm{loc}}\approx0\)，\(\eta_{\mathrm{geom}}\approx1.0\) → learned localization **几乎不增加**几何误差（在本 split 上）。

## 读数

1. **按冻结预注册**：¬G0 ⇒ `coverage_failure`；**禁止**放宽 FOV 门救 PASS。  
2. 机制上：O0G1b 路径 + U-Net 在可见帧上 **已证明** \(RGB\to\hat M\to p^{pose}\) 可工作，且远优于 mean。  
3. 与 O0R 对比：O0R 是 \(E_p\approx E_p^{B0}\)；本格 B2 为 \(0.05\times B0\)——**结构偏置有效**，但本 seed 的 coverage 合同未站稳。  
4. **O0C / O1 仍 LOCKED**（须 coverage 合格的 pose-supported 格）。  
5. 下一格若继续：新预注册（新 seed / 显式 coverage 诊断），**不是**改本格 G0 阈值。

## 条件性结论（不改 pattern）

\[
\boxed{
\textbf{conditional on visibility，}
RGB+D+\text{known geometry}
\textbf{ 可准确恢复 cup position（}B2\approx B1\text{）。}
}
\]

\[
\text{conditional accuracy}
\neq
\text{unconditional availability.}
\]

下一格 **O0V**（robust multi-view coverage）；**禁止**再赌 head-only seed。O0G2R/O0C/O1 LOCKED。

## Pattern

```text
G0 = FAIL (FOV=vis=0.902)
G1–G4 = PASS (descriptive; do not override primary)
pattern = coverage_failure
unlocks_o0c_prereg = false
unlocks_o1 = false
```
