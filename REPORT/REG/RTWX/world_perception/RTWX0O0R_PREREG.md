# RTWX-O0R 预注册 — Fresh Explicit Object-State Observability

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`object_pose_failure`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0R_REPORT.md`。  
依赖：O0D3 = `instrument_qualified`。  
**禁止**：换 camera / multi-view / 分辨率 / backbone / GAP / depth / seg；改 loss / pose 表示 / 阈值；改 O0–O0D3 ledger；跑 M2 / dynamics / attention / compute；R10；test 后再调超参。

## 科学问题

\[
\boxed{
\text{冻结视觉仪器是否能在 fresh episodes 上从 RGB 恢复 cup 的显式 }s^O\text{？}
}

\]

Primary：\(RGB_t\to(p_t,R_t)\)。Velocity secondary（失败可进 O1）。

## 冻结合同（= O0D3）

| 项 | 值 |
|----|------|
| 任务 | `place_empty_cup` |
| 相机 | **`head_camera` only** |
| RGB | **64×64** |
| Encoder | CoordConv + spatial flatten；**无 GAP** |
| seed | **21601**（与 O0/O0D* disjoint） |
| split | **48/24/24 × 120** |
| \(s^O\) | \((p,R,v,\omega)\)；\(R\) 四元数 |
| 训练 | train only；val = early-stop / checkpoint；test 只评一次 |

## Baselines

| ID | 定义 |
|----|------|
| B0 | train-mean \((\bar p,\bar R,\bar v,\bar\omega)\) |
| B1 | \(s_{t-1}^{O,\mathrm{GT}}\)（诊断；不进 claim） |
| **B2** | frozen visual instrument（primary） |

## Gates

- **G0**：test \(P_{\mathrm{FOV}}\ge0.95\) 且 \(P_{\mathrm{visible}}\ge0.90\)；失败 → `coverage_failure` STOP
- **G_pos**：\(E_p\le0.30\)
- **G_info**：\(E_p^{B2}\le0.85\,E_p^{B0}\)
- **G_ori**：median \(e_R\le15^\circ\) 且 \(P_{90}\le30^\circ\)（**full SO(3)**，primary）
- **G_vel**：\(E_v\le0.50\)，\(E_\omega\le0.60\)（不否决 pose）

Secondary（不可覆盖 primary）：symmetry-aware \(d_{SO(3)/G_{\mathrm{sym}}}\)（绕杯轴 yaw）。

## Patterns

| Pattern | 条件 | O1 |
|---------|------|-----|
| `coverage_failure` | ¬G0 | LOCKED |
| `object_pose_failure` | G0 ∧ (¬G_pos ∨ ¬G_info ∨ ¬G_ori) | LOCKED |
| `orientation_symmetry_limited` | G_pos∧G_info ∧ ¬G_ori ∧ symmetry secondary 过 | LOCKED（须新 state 定义） |
| `pose_supported_velocity_failed` | pose 全过 ∧ ¬G_vel | **unlock O1** |
| `explicit_object_pose_supported` | pose ∧ G_vel | **unlock O1** |

Pose supported := G0∧G_pos∧G_info∧G_ori。

## 明确不做

O1 filtering；object dynamics；M2；capacity；attention；adaptive compute。
