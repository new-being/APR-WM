# RTWX-O0D1 报告 — Perception Instrument Repair

日期：2026-08-29  
状态：**正式冻结** `scalar_memorization_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0D1_PREREG.md`  
产物：`runs/rtwx_o0d1/{run.json,overlays/,...}`  
**无科学 claim**；**不跑 O0R**；**不解锁 O1**。  
O0 / O0D 账本不变。

## 一句话

仪器仍未闭合。R1（32 张 \(p_x\) 过拟合）\(NRMSE=0.146>0.05\)。  
R3：front_camera 上杯子 **只有 \(1/3\) 帧在 FOV**（遮挡率 0；在画幅内时 sim id 与投影处 camera id **一致**）。  
**不**把 O0 读成 RGB→\(s^O\) 不可行。

## Gates

| Gate | 条件 | 结果 |
|------|------|------|
| R1 \(p_x\) mem | \(NRMSE<0.05\) | **失败**（0.146）；**不是**塌回均值 |
| R2 \(E_p\) mem | \(<0.10\) 或 \(\le0.2\times\) mean | **未跑**（R1 未过） |
| R3 visibility | \(P(\mathrm{vis})\ge0.9\) | **失败**（0.333） |

Pattern：**`scalar_memorization_failure`**。

## Visibility（A/B）

- Actor 表含 `021_cup` **id=66**。在 FOV 内：`id_uv` 与 sim id 一致；**`frac_occluded_in_fov=0`**。  
- **主失败是出画**：\(\mathrm{frac\_in\_fov}=0.333\)。O0D 的 median mask=0 不能再只归因于 id 撞错。  
- Overlay：`runs/rtwx_o0d1/overlays/`（投影十字 + 修复 mask）。

## 管线检查

| 检查 | 结果 |
|------|------|
| \(\mu,\sigma\)；\(\hat{\tilde p}=1\Rightarrow\mu+\sigma\) | **通过**（target std 均值 0、std=1） |
| 图像非退化 | median pixel std \(\approx 25\)；帧间 L2 大 |
| 梯度 / \(\Delta\theta\) | \(\|\nabla\|\neq0\)；\(\|\Delta\theta\|\approx2.78\)；46945 参数全 `requires_grad` |
| train loss | \(1.02\to 0.037\)（标准化 \(p_x\)） |
| 随机标签 32 张 | \(NRMSE=0.87\) **未记住** |

随机标签也没过，说明 **tiny CNN + 400 epoch 在 224 全图上仍不够把 32 个标量记死**；不是「预测被 evaluator 覆盖成 \(\mu\)」（`hat_equals_mean=false`）。

## 读数

1. **front_camera 不宜作为 O0 主合同**，除非换相机/裁剪/多视角并写成新 instrument（不是改 O0）。  
2. **O0R / O1 仍锁。** 下一修：更强过拟合仪器（更小 crop 或更多 epoch/容量）**并且**可见性合同，两路都过再开 O0R。  
3. 不讨论 ResNet vs ViT。
