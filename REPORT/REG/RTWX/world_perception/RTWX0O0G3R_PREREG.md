# RTWX-O0G3R 预注册 — Learned Canonical Correspondence + Kabsch

日期：2026-08-30  
状态：**已冻结（修订版）**  
依赖：O0G3=`oracle_orientation_geometry_supported`；O0G2R segmentation；O0C/O0C1 direct-R baseline  
**禁止**：改 position/fusion；扩大 direct-R 网络；ICP/RANSAC/learned fusion；rotation loss；arch sweep；解锁 O1。

## 科学问题

> \(RGBD\to\hat x^O\) 能否跨 fresh episode 泛化，并通过固定 Kabsch 恢复 \(R_{BO}\)？

\[
f_\theta(RGBD,\hat M)\to\hat x_i^O,\quad x_i^B\ \text{from depth}\Rightarrow\mathrm{Kabsch}\Rightarrow\hat R
\]

**仅**学习 \(\hat x^O\)；**无**最终 rotation loss。

## 冻结合同

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | **`head` + `observer`** |
| 输入 | RGB + depth（1ch，来自 \(x^B\) z）→ **4ch U-Net** → 3ch \(\hat x^O\) |
| 监督 | \(\tilde x^O=x^O/D_O\)；\(\mathcal L_{\mathrm{corr}}=\mathrm{L1}\) on GT mask pixels only |
| \(x^B\) | Position→world（64²） |
| 融合 | learned mask 内点并集 → **Kabsch only** |
| seeds | **29601 / 29602 / 29603**（运行前冻结） |
| 规模 | 每 seed **24/12/12 × 120** |

test 禁止 \(p^{GT},R^{GT},x^{O,GT}\)。

## Baselines

| ID | 定义 |
|----|------|
| B0 | 同 split **direct RGB→\(R\)**（CoordConv；dual pooled；对照 O0C mode） |
| B1 | GT mask + GT \(x^O\) → Kabsch（≈O0G3） |
| B2 | learned mask + learned \(\hat x^O\) → Kabsch（**主**） |

## Gates

| Gate | 条件 | 角色 |
|------|------|------|
| G0 | agg \(P(V^{any})\ge0.98\)；每 seed \(\ge0.95\)；足够点 \(\ge N_{\min}\) | STOP |
| G1 | train \(E_{\mathrm{corr,train}}^{norm}<0.05\) | instrument |
| G2 | test \(E_{\mathrm{corr}}^{norm}\) / P90 | **描述** |
| G3 | B2：med \(e_R\le15^\circ\)，**P90 \(\le30^\circ\)** | **primary** |
| G4 | \(f_{90}^{B2}\le0.5\,f_{90}^{B0}\) | confirmatory（弱则 tag） |
| G5 | \(\Delta P_{90},\Delta\) med vs B1 | 描述 |

Secondary（不扫 τ）：\(r_{\mathrm{fit}}\)、\(d_{\mathrm{view}}\)（per-view Kabsch）。

## Patterns

| Pattern | 条件 |
|---------|------|
| `coverage_failure` | ¬G0 |
| `correspondence_instrument_failure` | G0∧¬G1 |
| `geometry_mediated_orientation_failure` | G0∧G1∧¬G3 |
| `geometry_mediated_orientation_supported` | G0∧G1∧G3 |

G4 不过 → `mode_reduction_weak` tag，不改 primary pattern。

## 解锁

PASS → 允许预注册 **O0C2**（structured full \(T_{BO}\)）。**O1 LOCKED** until O0C2 PASS。
