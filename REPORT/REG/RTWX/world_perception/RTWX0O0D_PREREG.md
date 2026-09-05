# RTWX-O0D 预注册 — Object Perception Instrument Audit

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`perception_instrument_failure`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0D_REPORT.md`。  
依赖：O0 = `object_pose_failure`。  
**禁止**：解锁 O1；object dynamics；attention/compute；改 M2；改 O0 门限；R10；把本格写成「第三人称 RGB 对 \(s^O\) 不可观测」。

## 科学地位

O0 已冻结：

\[
\texttt{RTWX-O0 = object\_pose\_failure}
\]

B2 \(\approx\) B0 说明更像 **image\(\to\)pose 映射未建立**，而不是「映射已建立但不够准」。  
本格 **不产生 observability 论文结论**，只诊断失败来自 pipeline 还是视觉信息。

## 合同

| 项 | 冻结值 |
|----|--------|
| D1 数据 | **复用** `runs/rtwx_o0/cache_o0_train.npz`（不是 fresh confirmation） |
| D1 \(N\) | **256** 帧；无 augmentation；\(L=1\)；只预测 \((p,R)\) |
| D1 epochs | **200**；\(\mathrm{WD}=0\)；无 early-stop 到另一 split |
| D0/D2/D3 | 同任务/相机/224；oracle mask **只用于诊断**；seed **17601**；16/8/8 × 40 |
| Encoder | 与 O0 同结构（ResNet18 去 avgpool/fc + spatial flatten；**无 GAP**） |
| Velocity | **secondary**，不进入 D1/D2 损失 |

## 诊断问

- **D0**：\(RGB_t\) 与 \(s_t^O\) 是否同时；杯子是否在 FOV；pixel centroid 是否随 \(p_{xy}\) 变。
- **D1**：当前网络能否 **记住** 256 个 image\(\to\)pose。
- **D2**：\(RGB\odot M_{\mathrm{cup}}\)（全幅 canvas）\(\to(p,R)\)。
- **D3**：\(z_{2D}=(u_c,v_c,w,h,A)\to p\)（无神经网络）。

## Gates（诊断，不解锁 O1）

- **D1_ok**：\(E_p^{\mathrm{mem}}\le 0.15\)
- **D2_ok**：mask RGB 的 \(E_p\le 0.30\) 且 \(\le 0.85\times\) 该 audit 的 B0
- **D3_ok**：\(E_p^{z_{2D}}\le 0.30\)
- **D0_vis**：median visibility \(\ge 0.001\)（约 \(\ge 50\) px @ 224²）
- **D0_track**：\((u_c,v_c)\) 与 \((p_x,p_y)\) 的 max \(\lvert\rho\rvert\ge 0.25\)

## Patterns（仅仪器）

| Pattern | 条件 |
|---------|------|
| `perception_instrument_failure` | ¬D1_ok |
| `localization_bottleneck` | D1_ok 且 (D2_ok ∨ D3_ok) |
| `pose_geometry_unresolved` | D1_ok 且 ¬D2_ok 且 ¬D3_ok |

¬D0_vis 写入报告，**不**单独改写成不可观测。

## 明确不做

- 不换 ViT；不跑 O1；不谈 adaptive compute；不 fresh-confirm O0。
