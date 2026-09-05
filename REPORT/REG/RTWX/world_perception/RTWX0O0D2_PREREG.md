# RTWX-O0D2 预注册 — Perception Instrument Closure

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`image_memorization_failure`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0D2_REPORT.md`。  
依赖：O0D1 = `scalar_memorization_failure`。  
**禁止**：改 O0；解锁 O1；用本格数据当 O0 科学结论；换 ViT 比容量；R10。

## 账本

```text
O0   = object_pose_failure   (scientific reading DISABLED by instrument audit)
O0D  = perception_instrument_failure
O0D1 = scalar_memorization_failure
       front_camera FOV=0.333; id/occlusion OK when in FOV
O0D2 = this cell
O0R  = LOCKED until instrument_closed
O1   = LOCKED
```

O0 失败至少混合：**不合格 camera coverage** + **未闭合的 regression instrument**。

## 冻结合同

| 项 | 值 |
|----|------|
| seed | **19601** |
| D2-A | 32 index → embedding lookup → \(p_x\)；\(NRMSE<10^{-3}\)；随机标签同门 |
| D2-B | 32 张 **64×64 flatten MLP** → \(p_x\)；真标签与随机标签均 \(NRMSE<0.05\) |
| D2-C | GAP CNN vs CoordConv/flatten-map（无 GAP）；同 32 张真 \(p_x\) |
| FOV | 6×20；oracle \(p\) + 每相机 \(K,T\)；\(c^\star=\arg\max P_c\)，须 \(P\ge0.9\) |
| mem 图 | 复用 O0 train cache |

Flatten **不是**最终 perception 架构，只做 memorization diagnostic。

## Patterns（互斥，优先靠前）

| Pattern | 条件 |
|---------|------|
| `training_pipeline_failure` | ¬D2-A |
| `image_memorization_failure` | A ∧ ¬D2-B |
| `spatial_representation_failure` | A ∧ B ∧ spatial 过 ∧ GAP 不过 ∧ ¬(FOV 合同) |
| `coverage_insufficient` | A ∧ B ∧ \(\max_c P_c<0.9\)（可报 any-camera） |
| `instrument_closed` | A ∧ B ∧ \(\max_c P_c\ge0.9\) |

若 A∧B∧FOV 且 GAP 不过、spatial 过：记 **`instrument_closed`**，并在报告标 `spatial_bottleneck_on_GAP=true`。

仅 `instrument_closed` 允许另开 **O0R**（新 seed、新 observation contract）。本格不跑 O0R。
