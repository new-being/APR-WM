# RTWX-O0G2R 预注册 — Dual-View Geometry-Mediated Position Confirmation

日期：2026-08-29  
状态：**已冻结**  
依赖：O0V=`dual_view_coverage_supported`；O0G2 条件性 localization+geometry；O0G1b \(\delta_O\)  
**禁止**：改 O0G2/O0V；换/加相机；拟合 \(\delta_O\)；`RGB\to p`；混入 orientation 学习；解锁 O1（须先 O0C）。

## 科学问题

> 在预先验证的双视角覆盖合同下，冻结组件组合 \(RGBD\to\hat M\to p^{pose}\) 能否在 fully fresh 上稳定恢复 object position？

## 冻结合同

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | **`head` + `observer`**（O0V；不可换） |
| 网络 | O0G2 **small U-Net** family（shared；无 arch sweep） |
| \(\delta_O\) | `021_cup/model_data0.json:center*scale` |
| \(R_{BO}\) | **GT only**（reference correction nuisance；非本格 orientation 测试） |
| 融合 | \(V_t=\{c:\hat M_c\neq\emptyset\}\)；\(\mathcal P_B=\bigcup_c\mathcal P_B^{(c)}\)；**逐坐标 median** → \(p^{surf}\) → \(p^{pose}=p^{surf}-R_{BO}\delta_O\) |
| seed | **26601**；**48/24/24 × 120** |
| RGB | 64×64 |

## Baselines

| ID | 定义 |
|----|------|
| B0 | \(\bar p_{\mathrm{train}}\) |
| B1 | dual GT masks → fusion → \(p^{pose}\) |
| B2 | dual learned masks → fusion（**主**） |
| B3 | head-only learned（描述性；非主） |

## Gates

| Gate | 条件 |
|------|------|
| G0 | \(P(V^{any})\ge0.98\)（GT visibility） |
| G1 | B1：\(E_p\le0.20\)，med≤5 cm（报 strong≤2 cm） |
| G2 | 联合 \(P(\hat M_h\lor\hat M_o\neq\emptyset\mid V^{any})\ge0.95\)；另报每相机 IoU |
| G3 | B2：\(E_p\le0.20\)，med≤5 cm |
| G4 | \(E_p^{B2}\le0.85 E_p^{B0}\) |
| G5 | \(\eta=E_p^{B2}/E_p^{B1}\le1.5\) 或 \(\Delta E\le0.05\)（记 `localization_near_oracle`） |

## Patterns

| Pattern | 条件 |
|---------|------|
| `dual_view_coverage_failure` | ¬G0 |
| `oracle_fusion_geometry_failure` | G0∧¬G1 |
| `dual_view_localization_failure` | G1∧¬G2 |
| `dual_view_position_failure` | G2∧(¬G3∨¬G4) |
| `dual_view_geometry_position_supported` | G0–G4 |

## 解锁

PASS → 允许预注册 **O0C**（composite \(p+R\)，非-oracle \(R\)）。O1 LOCKED until O0C.
