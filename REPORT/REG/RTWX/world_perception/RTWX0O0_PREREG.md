# RTWX-O0 预注册 — Explicit Object-State Observability

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`object_pose_failure`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0_REPORT.md`。  
依赖：X0RGB1 = `spatial_perception_failure`（关闭 RGB→\(s^R\)）。  
**禁止**：跑 M2；object dynamics；task attention；adaptive compute；latent capacity；改 X0E–X0RGB1 ledger；R10；继续 RGB→\((q,\dot q)\)。

## 科学问题

\[
\boxed{
\text{第三人称 RGB 是否足以恢复 manipulation 所需的显式 object state }s^O\text{？}
}\]

\(s^R=(q,\dot q)\) **不从 RGB 估**（proprioception / oracle）。

## 冻结合同

| 项 | 冻结值 |
|----|--------|
| 任务 | `place_empty_cup` |
| 对象 | `env.cup` |
| \(s^O\) | \((p,R,v,\omega)\)，\(R\) 为四元数 |
| seed | **16601**；48/24/24 × 120 |
| 相机 | `front_camera`；**224×224** |
| 历史 | \(L=4\) |
| Encoder | ResNet18 + spatial feature + temporal head（ImageNet 初始化若可得，否则随机） |
| \(s^R\) | 采集记录但不进入 \(E_{\mathrm{vis}}\) |

不把 flatten MLP 作为 primary。对照：B0 train-mean pose；B1 上一帧 GT pose（诊断静态）。

## Gates（评分前冻结）

- **G0 coverage**：test 满额；位姿有限；train \(\mathrm{std}(\|p\|)\) 显示跨 episode 变化（\(\mathrm{std}(p_x)+\mathrm{std}(p_y)>0.02\) m）
- **G1 position**：\(E_p\le0.30\)（NRMSE，同 `_nrmse`）；并报告 median \(\|e_p\|\) (cm)
- **G2 orientation**：\(\mathrm{median}(e_R)\le15^\circ\)，\(P_{90}(e_R)\le30^\circ\)
- **G3 velocity**（不否决 pose）：\(E_v\le0.50\)，\(E_\omega\le0.60\)
- **G_info**：B2 的 \(E_p\) 与 \(\mathrm{median}(e_R)\) 均优于 B0（\(\le 0.85\times\) B0）

\(e_R=2\arccos(|q\cdot\hat q|)\)（度）。评价不换 NRMSE 定义。

## Patterns

| Pattern | 条件 |
|---------|------|
| `object_pose_failure` | ¬G0 或 ¬G1 或 ¬G2 或 ¬G_info |
| `pose_supported_velocity_failed` | G0∧G1∧G2∧G_info 且 ¬G3 |
| `explicit_object_state_supported` | G0∧G1∧G2∧G3∧G_info |

G1/G2 通过即解锁 O1（persistent filtering）；G3 失败只修速度/滤波，不关 pose 路线。

## 明确不做

- 不预测 RGB；不加 \(z_g\)；不改 M2；不开多物体 attention
