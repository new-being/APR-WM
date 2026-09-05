# RTWX-O0R 报告 — Fresh Explicit Object-State Observability

日期：2026-08-29  
状态：**正式冻结** `object_pose_failure`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0R_PREREG.md`  
产物：`runs/rtwx_o0r/{run.json,metrics.json,cache_o0r_*.npz}`  
仪器（冻结 O0D3）：**`head_camera` + 64×64 + CoordConv（无 GAP）**  
seed **21601**；48/24/24 × 120；**无 M2**；test 只评一次。

## 一句话

覆盖合同 **通过**（\(P_{\mathrm{FOV}}=P_{\mathrm{visible}}=0.962\)），朝向 primary **通过**（median \(9.3^\circ\)，\(P_{90}=14.7^\circ\)）。  
位置 **失败**：\(E_p=0.343>0.30\)，且相对 B0 **无信息增益**（\(0.343/0.350\approx0.98>0.85\)）。  
Pattern：**`object_pose_failure`**。**不解锁 O1**。

## Gates（阈值未放宽）

| Gate | 条件 | 结果 |
|------|------|------|
| G0 coverage | \(P_{\mathrm{FOV}}\ge0.95\)，\(P_{\mathrm{visible}}\ge0.90\) | **通过**（0.962 / 0.962） |
| G_pos | \(E_p\le0.30\) | **失败**（0.343） |
| G_info | \(E_p^{B2}\le0.85\,E_p^{B0}\) | **失败**（≈0.981×） |
| G_ori | median \(\le15^\circ\)，\(P_{90}\le30^\circ\) | **通过**（9.26° / 14.72°） |
| G_vel | \(E_v\le0.50\)，\(E_\omega\le0.60\) | **失败**（≈1.00） |

## 主表（test）

| 仪器 | \(E_p\) | median \(\|e_p\|\) (cm) | median \(e_R\) | \(P_{90}(e_R)\) | \(E_v\) | \(E_\omega\) |
|------|--------:|------------------------:|---------------:|----------------:|--------:|-------------:|
| B0 train-mean | 0.350 | 25.46 | \(11.46^\circ\) | \(14.39^\circ\) | 1.019 | 1.009 |
| B1 上一帧 GT | 0.011 | 0.00 | \(0.00^\circ\) | \(0.00^\circ\) | 1.266 | 1.220 |
| **B2 RGB（主）** | **0.343** | **25.62** | **\(9.26^\circ\)** | **\(14.72^\circ\)** | **1.004** | **1.001** |

Symmetry secondary：yaw-aligned median \(9.3^\circ\)（与 full-\(R\) 同量级）；**不覆盖** primary。`symmetry_limited=false`。

## 读数

1. **不是 coverage 失败。** O0D3 的 `head_camera` 合同在 fresh 数据上站得住。  
2. **不是「完全塌回均值朝向」。** G_ori 过；B2 median \(e_R\) 略优于 B0。  
3. **泛化失败在位置。** O0D3 上 train memorization \(E_p=0.003\)；本格 fresh \(E_p=0.343\approx\) B0。仪器能记住，**不能**把 \(p\) 泛化到新 episode。  
4. **O1 LOCKED。** 预注册：仅 pose-supported patterns 解锁 O1。  
5. **禁止**事后换 camera / backbone / 分辨率 / 放宽门限来救本格。下一格 **O0G**（几何分解）须新预注册。

## 描述性 audit（不改 pattern）

在 `cache_o0r_*.npz` 上复现 B2 后追加（2026-08-29）：

### Per-axis NRMSE（test）

| 仪器 | \(E_x\) | \(E_y\) | \(E_z\) | \(E_p\) |
|------|--------:|--------:|--------:|--------:|
| B0 | 1.124 | 0.812 | **0.068** | 0.350 |
| B2 | 1.117 | 0.815 | **0.067** | 0.348 |

**不是** \(E_z\gg E_x,E_y\)。\(z\equiv0.740\) m 无跨 episode 方差（桌面高度固定）；失败集中在 **\(x,y\)**，且 B2≈B0 逐轴重合 → 未学到可迁移的 image-space → world-\(xy\) 关系。

### Train/test 首帧 \(p_0\) 支持

| 轴 | train 范围 | test 范围 | test 超出 train min–max |
|----|------------|-----------|-------------------------|
| \(x\) | \([-0.29,0.30]\) | \([-0.29,0.30]\) | 0% |
| \(y\) | \([-0.20,0.05]\) | \([-0.18,0.05]\) | 4% |
| \(z\) | 0.740 | 0.740 | 0% |

无显著 **position-support extrapolation**。问题更接近：episode 内几乎静止（B1 \(E_p=0.011\)），模型需从单帧识别 **跨 episode 初始 \(xy\)**，CoordConv 记住了 train 帧但未泛化。

### 机制收束

\[
E_p^{\mathrm{mem}}(O0D3)=0.003
\quad\Rightarrow\quad
E_p^{\mathrm{fresh}}(O0R)=0.343\approx E_p^{B0}
\]

\[
\boxed{
\text{能记住 image}\rightarrow p\text{；未学到 }(u,v,\mathrm{app})\rightarrow(x,y,z)_{\mathrm{world}}\text{ 的可迁移几何。}}
\]

暂停 **whole-image RGB→\(p_B\)**。下一格 O0G：\(RGB\to(u,v,d)\to p_C\to p_B\)。
