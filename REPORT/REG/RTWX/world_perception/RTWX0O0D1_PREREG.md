# RTWX-O0D1 预注册 — Perception Instrument Repair

日期：2026-08-29  
状态：**已冻结**；**正式 RAN 2026-08-29**；**`scalar_memorization_failure`**。报告：`REPORT/REP/RTWX/world_perception/RTWX0O0D1_REPORT.md`。  
依赖：O0D = `perception_instrument_failure`。  
**禁止**：解锁 O1；把 O0 读成 RGB→\(s^O\) 不可行；换 ViT/ResNet 比容量；改 O0 门限；R10；未过本格就开 O0R。

## 地位

```text
O0   = object_pose_failure
O0D  = perception_instrument_failure
O0D1 = instrument repair / no scientific claim
O0R  = LOCKED until O0D1 全过（fresh confirmation，另格）
O1   = LOCKED until O0R PASS
```

目标只有两个：

\[
\text{R3: 杯子真的在图像里}\qquad
\text{R1+R2: 网络能记住 image}\rightarrow\text{pose}
\]

## 冻结合同

| 项 | 值 |
|----|------|
| seed | **18601** |
| vis collect | 6 ep × 20 step；`front_camera`；同 `place_empty_cup` |
| R1 | 32 张图；tiny CNN；只预测 \(p_x\)；无 aug / WD / dropout / early-stop / val |
| R2 | R1 通过后；256 张；tiny CNN → \(p\in\mathbb{R}^3\)；复用 O0 train cache |
| 数据 | vis 新采；mem 用 `runs/rtwx_o0/cache_o0_train.npz` |

## Gates

- **R1** \(NRMSE_{\mathrm{train}}(p_x)<0.05\)
- **R2** \(E_p^{\mathrm{train}}<0.10\) **或** \(\le 0.2\times E_p^{\mathrm{mean}}\)
- **R3** \(P(\text{cup visible})\ge 0.9\)  
  visible := 投影在 FOV 且（修复后 actor-id mask 面积 \(\ge 50\) px **或** 投影像素 actor-id 非空）

## 必做检查（评分前）

A. `id_cup^{sim}` vs 投影处 `id^{camera}`；segmentation 非零 id ↔ actor name 表  
B. \(T_{CB},K\) 投影 \(z_C>0\) 且 \(u,v\) 在幅面内（id 错 / 遮挡 / 出画分列）  
C. overlay：投影中心 + mask + id + \(p\)  
D. \(\mu,\sigma\) 反标准化 smoke：\(\hat{\tilde p}=1\Rightarrow\hat p=\mu+\sigma\)  
E. 32 张随机标签记忆  
F. \(\|\nabla L\|\)、\(\|\Delta\theta\|\)、`requires_grad`  
G. 图像 \(\mathrm{std}\) 与帧间差

## Patterns

| Pattern | 条件 |
|---------|------|
| `scalar_memorization_failure` | ¬R1 |
| `pose_memorization_failure` | R1 ∧ ¬R2 |
| `visibility_contract_failure` | R1 ∧ R2 ∧ ¬R3 |
| `instrument_repair_supported` | R1 ∧ R2 ∧ R3 |

全过才允许另开 **O0R**（新 seed、fresh 数据）。本格不跑 O0R。
