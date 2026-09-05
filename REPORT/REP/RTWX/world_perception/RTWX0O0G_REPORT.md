# RTWX-O0G 报告 — Geometry-Mediated Object Position（G0/G1）

日期：2026-08-29  
状态：**正式冻结** `geometry_insufficient`（G0 PASS ∧ G1 FAIL）  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0G_PREREG.md`  
产物：`runs/rtwx_o0g/{run.json,metrics.json,cache_o0g_geom.npz,run.log}`  
相机：**`head_camera`**；seed **22601**；8×32；stage **g01**  
**不改 O0R**；**不解锁 O1**；**G2 未跑**（按预注册 G1 失败则 STOP）。

## 一句话

Oracle 投影几何 **通过**（数值精度）。Oracle mask + depth centroid 相对 cup `pose.p` **未过** 5 cm 门（median \(5.57\,\mathrm{cm}\)），pattern：

\[
\boxed{\texttt{geometry\_insufficient}}
\]

失败主因不是相机合同坏掉，而是 **pose 原点 vs 可见表面质心** 的系统 \(z\) 偏置（约 \(+5.1\,\mathrm{cm}\)）。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 oracle \((u,v,z_C)\to p_B\) | \(E_p\le0.05\)，median \(\le5\,\mathrm{mm}\) | **PASS**（\(E_p\sim10^{-16}\)，median \(\sim10^{-16}\,\mathrm{m}\)） |
| G1 oracle mask+depth centroid | \(E_p\le0.20\)，median \(\le5\,\mathrm{cm}\) | **FAIL**（\(E_p=0.072\)，median \(5.57\,\mathrm{cm}\)） |
| G2 learned \(\hat M\) | — | **未跑** |

## 主表

| 层 | \(E_p\) | median \(\|e\|\) | P90 | n |
|----|--------:|-----------------:|----:|--:|
| G0 | \(1.3\times10^{-16}\) | \(\sim0\) | \(\sim0\) | 256 |
| G1 | 0.072 | 5.57 cm | 5.81 cm | 254/256 valid |

## 描述性 audit（不改 pattern / 不放宽门限）

G1 误差分解（相对 `env.cup.get_pose().p`）：

| | \(x\) | \(y\) | \(z\) |
|--|------:|------:|------:|
| bias (cm) | −0.38 | +0.22 | **+5.11** |
| xy-only median \(\|e\|\) | | **1.39 cm** | |
| 去 bias 后 median \(\|e\|\) | | **1.29 cm** | |

\[
\boxed{
\text{G1 几乎是固定 }+5\,\mathrm{cm}\text{ 的 }z\text{ 帧差（表面质心 vs pose 原点），不是 }xy\text{ 几何崩坏。}}
\]

G0 已排除 `camera_geometry_contract_failure`。

## 读数

1. **已知 \(K,T_{BC}\) 合同正确** — 不必再怀疑外参/内参/帧约定导致 O0R 位置失败。  
2. **传感器 3D 在 localization 已知时大体可用** — 但当前 G1 目标是 `pose.p`，与 mask 表面质心不对齐。  
3. **按冻结门限** → `geometry_insufficient`；**不自动开 G2**。  
4. 若继续：建议新预注册 **G1b / 目标重定义**（例如相对表面质心、或已知 cup 半高校正），再开 G2；**禁止**回改本格门限救 PASS。

## 后续（只读指针）

O0G1b（2026-08-29）已用资产 \(\delta_O\)（非本格 test bias）对齐 → `geometry_reference_aligned`。见 `RTWX0O0G1B_REPORT.md`。**本格 pattern 不改。**

## Pattern

```text
G0 = PASS  (camera geometry contract OK)
G1 = FAIL  (median 5.57cm > 5cm; dominated by +5.1cm z bias)
G2 = STOP
pattern = geometry_insufficient
unlocks_o1 = false
```
