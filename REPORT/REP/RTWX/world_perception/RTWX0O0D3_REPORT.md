# RTWX-O0D3 报告 — Spatial Perception Instrument Qualification

日期：2026-08-29  
状态：**正式冻结** `instrument_qualified`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0D3_PREREG.md`  
产物：`runs/rtwx_o0d3/{run.json,metrics.json}`  
**无科学 claim**（train memorization，非泛化）；**本格不跑 O0R**；**不解锁 O1**。

## 一句话

冻结合同 **`head_camera` + CoordConv（无 GAP）** 在 256 帧新采数据上 **全部过门**。  
允许另开 **O0R**（fresh confirmation）；仍不能解读 O0 为 RGB→\(s^O\) 一般性 no-go。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| G0 coverage | \(P_{\mathrm{FOV}}\ge0.95\)，\(P_{\mathrm{visible}}\ge0.90\) | **过**（均为 **1.000**） |
| G1 \(p_x\) | \(NRMSE<0.05\) | **过**（**0.0087**） |
| G2 随机标量 | \(NRMSE<0.05\) | **过**（**0.00097**） |
| G3 position | \(E_p<0.10\) 或 \(\le0.2\times\) mean | **过**（\(E_p=\)**0.0032**，mean 0.253） |
| G4 rotation | median \(<5^\circ\)，\(P_{90}<15^\circ\) | **过**（**0.16°** / **0.28°**） |

Pattern：**`instrument_qualified`**。`unlocks_o0r=true`。

## 合同（冻结）

- 任务：`place_empty_cup`（不换任务）
- 相机：**`head_camera` only**（非 O0 的 `front_camera`）
- RGB：**64×64**；CoordConv + spatial flatten；**无 GAP**
- collect：seed **20601**；8×32 = **256** 帧（instrument collection，非 O0 数据）

## 对称性 audit（G4）

完整 \(R\) 在 train memorization 上过门。绕杯轴 yaw 对齐后 residual 约 **10°**（未达 symmetry 替代门），但 **原始 G4 已过**，故 pattern 仍为 `instrument_qualified`。  
O0R 预注册应明确：若轴对称不可辨识，\(s^O\) 可能需 \(SE(3)/G_{\mathrm{sym}}\) 而非机械 SE(3)——**不能事后改 O0**。

## 机制链（相对 O0）

| 阻塞 | O0D 诊断 | O0D3 结论 |
|------|----------|-----------|
| `front_camera` FOV≈0.33 | O0D1 | **`head_camera` FOV=1** |
| GAP 抹位置 | O0D2 CoordConv 0.011 | G1–G3 全过 |
| flatten 随机门 | O0D2 B 失败 | **本格不用 flatten** |

O0 的 `object_pose_failure` **保留**；科学解读仍 **DISABLED** 直到 O0R 在 fresh 数据上过 O0 pose gates。

## 下一格

**O0R**（另写预注册）：`head_camera` + 冻结 CoordConv；fresh seed；48/24/24；测 **泛化**，不是 memorization。  
**O1** 仍 LOCKED 直到 O0R PASS。
