# RTWX-O0D 报告 — Object Perception Instrument Audit

日期：2026-08-29  
状态：**正式冻结** `perception_instrument_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0D_PREREG.md`  
产物：`runs/rtwx_o0d/{run.json,metrics.json,overlays/,cache_o0d_*.npz}`  
**无科学 claim**；**不解锁 O1**；不改 O0 门限。  
O0 仍为 `object_pose_failure`。

## 一句话

当前 pipeline **连 256 张 O0 训练帧都记不住**（D1 \(E_p=0.645=\) 子集 train-mean）。  
这是 **仪器/映射未建立**，不是「已经学会 pose 但泛化不够」。  
不据此声称 RGB 对 \(s^O\) 不可观测。

## 诊断表

| 门 | 结果 | 数 |
|----|------|-----|
| D0_vis | **失败** | median vis \(=0\)；仅 \(40\%\) 帧 mask 非空 |
| D0_track | **通过** | \(\lvert\rho(u,p_x)\rvert=0.80\)（有像素的子集） |
| D1_ok | **失败** | \(E_p^{\mathrm{mem}}=0.645\)，B0 \(=0.645\)；median \(24.7\) cm |
| D2_ok | **失败** | mask RGB \(E_p=0.381>\) B0 \(0.331\) |
| D3_ok | **失败** | \(E_p=2.20\)（**不可解释**：多数帧 mask 空，\(z_{2D}\) 退化） |

Pattern：**`perception_instrument_failure`**（¬D1）。

## D1（主结论）

复用 O0 train cache，256 帧，\(L=1\)，200 epoch，WD=0，无 augmentation。  
与该 256 帧的 train-mean **重合到 \(10^{-4}\) 量级**。  
ResNet18 足以记住 256 个 pair；做不到 ⇒ **loss / 标签 / 图像管线 / 输出头** 之一坏了，禁止据此换更大 backbone，也禁止谈 observability。

## D0（可见性）

32 张 overlay：`runs/rtwx_o0d/overlays/`。  
RGB 与 \(s^O\) 在同一 `take_action` 前记录。  
**median 可见面积为 0**：oracle actor-id mask 在 224 上经常对不上杯子（id 通道、遮挡、或杯子不在 FOV）。  
D3 在此条件下 **不能** 用来否定几何可观测性。

## 读数

1. **先修 instrument**（mask id、FOV、D1 过拟合），再谈 detector/PnP。  
2. **O1 / D0 object dynamics / attention 仍锁。**  
3. D0_track 通过只说明：一旦 mask 非空，像素位置与 \(p_x\) 有相关；不是 D1 通过。
