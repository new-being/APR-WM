# RTWX-O0G 预注册 — Geometry-Mediated Object Position

日期：2026-08-29  
状态：**已冻结（诊断/仪器格；非 O0R 重跑）**  
依赖：O0R = `object_pose_failure`（coverage/ori 过；\(xy\) 泛化失败）。  
**禁止**：改 O0R pattern/门限；继续 whole-image RGB→\(p_B\)；换 camera（仍用 **`head_camera`**）；解锁 O1；M2；R10。

## 科学问题

> 若不让网络直接从整张 RGB 回归 world \(p\)，而先恢复 image-space localization/depth，再经 **已知 camera 几何** 变换，位置泛化是否恢复？

\[
RGB\to(u,v,d)\to p_C\to p_B,\qquad p_C=dK^{-1}[u,v,1]^T,\qquad p_B=T_{BC}p_C
\]

已知投影/外参 **不由网络重学**（APR-WM：已知结构不让网络重学）。

## 冻结合同

| 项 | 值 |
|----|------|
| 任务 | `place_empty_cup` |
| 相机 | **`head_camera`**（O0R 合同） |
| seed | **22601**（与 O0/O0D*/O0R disjoint） |
| collect | 8×32 诊断集 + 正式格另定 |

## 三层诊断（顺序执行）

### G0 — Oracle \((u,v,d)\to p_B\)

Simulator 投影中心 + GT depth（或 \(z_C\)）。经 \(K,T_{BC}\) 恢复 \(\hat p_B\)。  
要求 \(E_p\ll0.05\)（近数值精度）。  
失败 → **`camera_geometry_contract_failure`** STOP。

### G1 — Oracle mask + depth → \(p_B\)

GT cup mask；masked depth / 3D 点云 robust centroid → \(\hat p_B\)。  
回答：localization 已知时，传感器几何是否足够？  
过 → **`geometry_feasible`**。

### G2 — Learned \(\hat M\) + geometry

\(RGB\to\hat M_{\mathrm{cup}}\)；结合 depth/几何 → \(\hat p\)。  
**本格 primary 视觉 instrument**（若 G0/G1 过）。

## Patterns（本格）

| Pattern | 条件 |
|---------|------|
| `camera_geometry_contract_failure` | ¬G0 |
| `geometry_feasible_localization_pending` | G0∧G1∧¬G2 |
| `geometry_mediated_position_supported` | G0∧G1∧G2 过 fresh 泛化门（另写于 G2 正式合同） |
| `geometry_insufficient` | G0 过 ∧ G1 失败 |

O1 仍 LOCKED（须 position 科学格 PASS，不是 G0/G1 oracle）。

## 明确不做

- 不 rescale CNN/ViT whole-image \(p\) 回归  
- 不改 O0R  
- 不在本格 claim RGB→\(s^O\) 最终成功

## O0R audit 输入（只读）

- \(E_x,E_y\gg E_z\)；\(z\) 常数  
- 非 support extrapolation  
- 见 `RTWX0O0R_REPORT.md` audit 节
