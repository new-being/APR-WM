# RTWX-O0D3 预注册 — Spatial Perception Instrument Qualification

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`instrument_qualified`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0D3_REPORT.md`。  
依赖：O0D2 = `image_memorization_failure`（CoordConv \(p_x=0.011\)；`head_camera` \(P_{\mathrm{FOV}}=1\)）。  
**禁止**：改 O0 / O0D2 pattern；救 flatten gate；扫 camera/backbone/分辨率；换任务；开 O0R（须本格 `instrument_qualified`）；R10；解锁 O1。

## 地位

不是救 D2，而是资格审查 **新 observation/perception contract**：

\[
\texttt{head\_camera}
+
\text{CoordConv / spatial map，无 GAP}
\]

任务仍为 `place_empty_cup`。Flatten 不再进入本格门。

## 冻结合同

| 项 | 值 |
|----|------|
| seed | **20601** |
| 相机 | **`head_camera` only** |
| RGB | **64×64**（与 O0D2 CoordConv 诊断同分辨率） |
| Encoder | CoordConv + flatten spatial map；**禁止 GAP** |
| collect | 8 ep × 32 step（256 帧） |
| G1/G2 | 32 帧 \(p_x\) / 随机标量 |
| G3/G4 | 256 帧 train memorization（非 test 泛化） |

## Gates

- **G0** \(P_{\mathrm{FOV}}\ge0.95\) 且 \(P_{\mathrm{visible}}\ge0.90\)
- **G1** \(NRMSE(p_x)<0.05\)
- **G2** 随机标签 \(NRMSE<0.05\)
- **G3** \(E_p^{\mathrm{train}}<0.10\) 或 \(\le0.2\times E_p^{\mathrm{mean}}\)
- **G4** median \(e_R<5^\circ\) 且 \(P_{90}<15^\circ\)  
  若失败：做 **轴对称 audit**（绕杯轴 yaw）。若去 yaw 后残差过门 → `rotation_symmetry_limited`，**不是** perception failure。

## Pattern

| Pattern | 条件 |
|---------|------|
| `coverage_failure` | ¬G0 |
| `scalar_spatial_failure` | G0 ∧ ¬G1 |
| `capacity_control_failure` | G0∧G1 ∧ ¬G2 |
| `position_memorization_failure` | G0–G2 ∧ ¬G3 |
| `rotation_memorization_failure` | G0–G3 ∧ ¬G4 ∧ ¬symmetry |
| `rotation_symmetry_limited` | G0–G3 ∧ ¬G4 ∧ symmetry |
| `instrument_qualified` | G0∧G1∧G2∧G3 ∧ (G4 ∨ symmetry contract) |

仅 `instrument_qualified` 允许另开 **O0R**（fresh 48/24/24，本格不跑）。
