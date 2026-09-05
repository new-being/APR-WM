# RTWX-O0G2 预注册 — Learned Localization + Geometry-Mediated Position

日期：2026-08-29  
状态：**已冻结**  
依赖：O0G1b = `geometry_reference_aligned`（known \(\delta_O\)；median 1.65 cm）  
**禁止**：`RGB\to p_B` 直接回归；改 O0G/O0G1b/O0R；拟合新 reference offset；换 camera/task；解锁 O1；跑 O0C；architecture sweep。

## 科学问题

> 学习系统是否**只**需从 RGB 学会「杯子在哪里」，而把 metric 3D recovery 交给 depth + 已知几何？

\[
RGB\to\hat M_O
\to(\hat M_O,D,K,T_{BC})\to\hat p^{\mathrm{surf}}
\to\hat p^{\mathrm{pose}}=\hat p^{\mathrm{surf}}-R_{BO}\delta_O
\]

## 冻结合同

| 项 | 值 |
|----|------|
| 任务 / 相机 | `place_empty_cup` / **`head_camera`** |
| 观测 | RGB **64×64** + depth/Position→xyz map 64 |
| \(\delta_O\) | O0G1b 冻结：`021_cup` model_data0 `center*scale` |
| 网络 | **small U-Net**（仅 binary cup mask；无 arch sweep） |
| seed | **24601** |
| 数据 | **48 / 24 / 24 × 120** |
| 监督 | oracle mask 仅 train；test **永不**输入 GT mask |

## Baselines

| ID | 定义 |
|----|------|
| B0 | \(\bar p_{\mathrm{train}}\) |
| B1 | \(M^{GT}+D\to p^{\mathrm{surf}}\to p^{\mathrm{pose}}\)（oracle ceiling） |
| B2 | \(RGB\to\hat M+D\to\ldots\)（**主**） |

## Gates

| Gate | 条件 |
|------|------|
| G0 coverage | \(P_{\mathrm{FOV}}\ge0.95\)，\(P_{\mathrm{visible}}\ge0.90\) |
| G1 oracle | B1：\(E_p\le0.20\)，median \(\le5\,\mathrm{cm}\)（另报 strong \(\le2\,\mathrm{cm}\)） |
| G2 loc | median IoU \(\ge0.50\)；\(P(\emptyset\mid\mathrm{vis})\le0.05\) |
| G3 pose | B2：\(E_p\le0.20\)，median \(\le5\,\mathrm{cm}\) |
| G4 info | \(E_p^{B2}\le0.85\,E_p^{B0}\) |

空预测帧计入失败（无 selection bias）。

## Patterns（仅此五种）

| Pattern | 条件 |
|---------|------|
| `coverage_failure` | ¬G0 |
| `oracle_geometry_failure` | G0 ∧ ¬G1 |
| `learned_localization_failure` | G0∧G1 ∧ ¬G2 |
| `localization_geometry_failure` | G0∧G1∧G2 ∧ (¬G3 ∨ ¬G4) |
| `geometry_mediated_position_supported` | G0–G4 全过 |

## 误差分解（描述性）

\[
\Delta E_{\mathrm{loc}}=E_p^{B2}-E_p^{B1},\qquad
\eta_{\mathrm{geom}}=E_p^{B2}/E_p^{B1}
\]

## 解锁

- 本格 PASS → 允许预注册 **O0C**（composite \(p+R\)）；**不**直接解锁 O1  
- O1 LOCKED until O0C pose-supported
