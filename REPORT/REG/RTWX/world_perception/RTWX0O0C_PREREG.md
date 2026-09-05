# RTWX-O0C 预注册 — Composite Non-Oracle Object Pose Confirmation

日期：2026-08-29  
状态：**已冻结**  
依赖：O0G2R=`dual_view_geometry_position_supported`；O0R orientation gate（med 9.3°）；O0V 双视角合同；O0G1b \(\delta_O\)  
**禁止**：改 O0G2R/O0V/O0R 门限；换/加相机；拟合 \(\delta_O\)；arch sweep；`RGB\to p`；neural fusion；解锁 O1 除非本格 PASS。

## 科学问题

> 去掉 \(R_{BO}^{GT}\) 后，视觉系统能否在 fresh 上联合恢复完整 \(T_{BO}=(R_{BO},p_{BO})\)？

状态定义：

\[
s_t^{O,\mathrm{pose}}=T_{BO,t}=(R_{BO,t},p_{BO,t})
\]

\(O\) = asset cup **pose origin**（非 surface / mesh AABB 混写）。

## 冻结合同

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | **`head` + `observer`**（O0V；position **与** orientation 统一） |
| Localization | O0G2 small U-Net shared |
| Position | union masked points → **per-axis median** \(p^{surf}\) → \(p^{pose}=p^{surf}-\hat R_{BO}\delta_O\) |
| \(\delta_O\) | `021_cup/model_data0.json:center*scale`（禁止重拟合） |
| Orientation | CoordConv family；\(RGB^{(c)}\to\hat R_{CO}^{(c)}\)；\(\hat R_{BO}^{(c)}=R_{BC}^{(c)}\hat R_{CO}^{(c)}\) |
| \(R\) 融合 | 有效视角 **SO(3) GeoMedian / chordal SVD mean**（无 neural fusion） |
| seeds | **27601 / 27602 / 27603** |
| 规模 | 每 seed **24/12/12 × 120** |
| RGB | 64×64 |

## Baselines

| ID | 定义 |
|----|------|
| B0 | \((\bar p,\bar R)_{\mathrm{train}}\) |
| B1 | GT masks + \(R^{GT}\) → \(p^{pose}\)（O0G2R ceiling；\(E_p^{GT\text{-}R}\)） |
| B2 | GT masks + \(\hat R\) → \(T_{BO}\)（隔离 \(R\to p\) 传播） |
| B3 | learned masks + \(\hat R\) → \(T_{BO}\)（**主**） |

## Gates

| Gate | 条件 |
|------|------|
| G0 | agg \(P(V^{any})\ge0.98\)；每 seed \(\ge0.95\)；联合 \(P(\hat M^{any}\neq\emptyset\mid V^{any})\ge0.95\) |
| G1 | B3：\(\mathrm{median}\,e_R\le15^\circ\)，\(P_{90}\le30^\circ\) |
| G2 | B3：\(E_p\le0.20\)，med \(\|e_p\|\le5\) cm（报 strong≤2 cm） |
| G3 | \(E_p^{B3}\le0.85 E_p^{B0}\)；secondary \(e_R^{B3}<e_R^{B0}\) |
| G4 | \(\eta_{\mathrm{pose}}=E_p^{B3}/E_p^{B1}\le1.5\) 或 \(\Delta E_p\le0.05\) → `near_oracle_position`（非硬 FAIL） |

另报：\(\Delta E_R=E_p^{B2}-E_p^{B1}\)（orientation→position 传播代价）。

每 seed：不得 \(E_p^{B3}\approx E_p^{B0}\)（要求 \(E_p^{B3}\le0.85 E_p^{B0}\) 于该 seed）。

## Patterns

| Pattern | 条件 |
|---------|------|
| `coverage_failure` | ¬G0 |
| `orientation_failure` | G0 ∧ ¬G1 |
| `orientation_position_interface_failure` | G1 ∧ ¬G2 |
| `composite_pose_failure` | G1∧G2 ∧ ¬G3 |
| `composite_object_pose_supported` | G0–G3 |

## 解锁

PASS → **解锁 O1** 预注册/实现（persistent filtering）。  
本格不要求单帧 \(v,\omega\)。
