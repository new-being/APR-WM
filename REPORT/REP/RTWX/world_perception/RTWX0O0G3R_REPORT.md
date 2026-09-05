# RTWX-O0G3R 报告 — Learned Canonical Correspondence + Kabsch

日期：2026-08-30  
状态：**正式冻结** `coverage_failure`（G0 STOP）  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G3R_PREREG.md`  
产物：`runs/rtwx_o0g3r/{summary.json,run.json,metrics.json,cache_o0g3r_s*,run.log}`  
合同：`head+observer`；RGBD 4ch U-Net → \(\hat x^O/D_O\) + 固定 Kabsch；seeds **29601/02/03**；24/12/12×120；B0/B1/B2  
**不改** position/fusion；**无** rotation loss；**无** O0C2 解锁；O1 LOCKED。

## 一句话

双视角数据合同在 **足够对应点帧占比** 与 **learned joint detection** 上未达 G0；primary scientific gate **未进入**。  
仪器侧：correspondence learner **未拟合**（train \(E_{\mathrm{corr}}^{\mathrm{norm}}\approx0.52\)）；oracle B1 仍近零误差，说明 Kabsch 管线完好。

\[
\boxed{\texttt{coverage\_failure}}
\]

## 运行注记

- 首次跑：RoboTwin 采集完成后 GPU 未释放 → mask U-Net 训练 OOM（10:22）。
- 修复：采集后 `_release_cuda()`；自 cache 重启（11:26）→ 12:16 完成。
- 9 个 split cache 均复用，无重采。

## Gates（按预注册互斥链）

| Gate | 条件 | 结果 | 备注 |
|------|------|------|------|
| **G0** | \(P(V^{any})\ge0.98\)；每 seed \(\ge0.95\)；\(P_{\mathrm{enough}}\ge0.95\)；joint det \(\ge0.95\) | **FAIL** | any=**1.000** ✓；joint=**0.941** ✗；\(P_{\mathrm{enough}}\)=**0.875** ✗ |
| G1 | train \(E_{\mathrm{corr}}^{\mathrm{norm}}<0.05\) | FAIL（0.517） | instrument；pattern 已在 G0 终止 |
| G2 | test \(E_{\mathrm{corr}}^{\mathrm{norm}}\) | med **0.523**，P90 **0.718** | 描述性 |
| G3 | B2 med \(\le15°\)，P90 \(\le30°\) | FAIL | med **143.8°**，P90 **176.0°** |
| G4 | \(f_{90}^{B2}\le0.5 f_{90}^{B0}\) | FAIL | 0.069 vs 0.081×0.5 |

**Pattern 判定**：\(\neg G0 \Rightarrow\) `coverage_failure`（STOP；不将 G3 作 primary 科学结论）。

## G0 明细

| 指标 | 聚合 | 门限 |
|------|------|------|
| \(P(V^{any})\) | **1.000** | \(\ge0.98\) |
| joint detection（visible 帧上 learned mask 任一侧命中） | **0.941** | \(\ge0.95\) |
| \(P_{\mathrm{enough}}\)（visible 且 GT 对应点 \(\ge N_{\min}=50\)） | **0.875** | \(\ge0.95\) |

| seed | \(P(V^{any})\) | \(P_{\mathrm{enough}}\) |
|------|---------------:|------------------------:|
| 29601 | 1.000 | 0.842 |
| 29602 | 1.000 | 0.872 |
| 29603 | 1.000 | 0.912 |

可见性满分，但 **双视角并集对应点充足率** 与 **joint det** 均未过线——本格在 primary 链上于 coverage 即停止。

## Baselines（test，描述性；G0 STOP 后仍记录）

| | med \(e_R\) | P90 | \(f_{90}\) | \(E_{\mathrm{corr}}^{\mathrm{norm}}\) med | \(r_{\mathrm{fit}}\) med | \(d_{\mathrm{view}}\) med |
|--|------------:|----:|-----------:|--------------------------------------------:|--------------------------:|----------------------------:|
| **B0** direct RGB→\(R\) | **6.0°** | 95.7° | 0.081 | — | — | 0.0° |
| **B1** oracle \(x^O\) + Kabsch | **0.0°** | **0.0°** | 0.000 | — | ~0 | 0.0° |
| **B2** learned \(\hat x^O\) + Kabsch | 143.8° | 176.0° | 0.069 | 0.523 | 0.040 m | 169.8° |

- **B1 ≈ O0G3**：oracle correspondence + Kabsch 仍 **med/P90=0** → 几何天花板未被本格推翻。
- **B2**：\(\hat x^O\) 误差 ~0.5\(D_O\) 量级 → Kabsch 输出接近随机旋转；\(d_{\mathrm{view}}\) 极大（双视角独立 Kabsch 严重不一致）。
- **B0**：中位数尚可、**P90 仍 ~96°**（O0C 尾部模式仍在）；本 split 上 B0 中位优于 B2，但 B2 未过 instrument，不宜作机制对照。

## 训练仪器

| 模块 | 结果 |
|------|------|
| Mask U-Net（冻结 family） | val loss → **0**；test median IoU head/obs = **1.0** |
| RGBD corr U-Net | val L1 ~**0.51**（25 ep 内无实质下降）；G1 **FAIL** |
| B0 CoordConv | 已训；见上表 |

## G5 Oracle gap（描述）

\[
\Delta_{\mathrm{med}} = 143.8° - 0°,\quad \Delta_{P90} = 176.0° - 0°
\]

在 B1 近零前提下，B2 的 orientation 误差 **几乎全部可归因于 learned correspondence 成本**（本跑中 learner 未过 instrument）。

## 读数

1. **Primary**：G0 未 PASS → 按预注册 **STOP**，pattern = `coverage_failure`；**不**解锁 O0C2，**不**宣称 geometry-mediated orientation 已测。
2. **仪器**：correspondence U-Net 在 25 epoch 内未将 train \(E_{\mathrm{corr}}^{\mathrm{norm}}\) 压至 0.05 以下；即使忽略 G0，亦会落入 `correspondence_instrument_failure`。
3. **机制对照仍成立**：B1 完美 + B2 崩溃 → 与 O0G3 一致，瓶颈在 **\(\hat x^O\) 学习**，不在 Kabsch。
4. **不宜过度解读 G3/G4**：G0 STOP 下 G3/G4 仅为描述性记录；B0 vs B2 的 \(f_{90}\) 不可作 confirmatory 结论。
5. **O1 LOCKED**；未预注册 O0C2。

## Pattern

```text
pattern = coverage_failure
G0_fail: P_enough=0.875 (<0.95), joint_det=0.941 (<0.95)
G1_fail_descriptive: E_corr_train_norm=0.517
unlocks_o0c2_prereg = false
unlocks_o1 = false
mode_reduction_weak = false
```
