# RTWX-O0E0P0 预注册 — Controlled-Pose Perception Qualification

日期：2026-08-31  
状态：**已冻结 / RAN** `controlled_pose_observation_qualified`  
依赖：O0E0=`observation_support_failure`（**distribution insufficient**；B2 **UNTESTED**）  
**禁止**：改 O0E0 / O0V / O0G2R 门；开 B0/B1/B2 science；加第三相机；用自由落体/dynamics 产生 pose；把本格写成 natural-task axis 必要；改 Fibonacci/B2 搜索；开 O0E1 / O1。

## 科学问题（仅 instrument）

> 不改变 estimator、不把 GT 喂给 estimator 的前提下，能否获得一套 **observation support 充分** 且 **axis DOF 真正被激励** 的合法视觉数据？

\[
\boxed{G_{\mathrm{support}}+G_{\mathrm{excitation}}}
\]

本格 **不评价** \(RGBD\to n\)。B2 保持冻结、未跑。

### 措辞（硬）

| 本格最多证明 | 本格不能证明 |
|--------------|--------------|
| 受控多姿态下 **可以合法检验** effective axis | 自然 `place_empty_cup` 需要/支持 axis estimation |
| 视觉可观测性数据合同 | 这些状态由自然 dynamics 产生 |

自然任务 \(n\approx e_z\) 是 **task prior**，与 representation capability 分开。

## 冻结合同

| 项 | 值 |
|----|------|
| 类型 | **perception-only static render**：`set_pose` → RGB-D；**不** physics rollout |
| 相机 | **head + observer**（与 O0V/O0E0 同；P0 内不加相机） |
| RGB | **128²**（与 O0E0 同） |
| seed | **37601** |
| 规模 | train **250** / val **100** / test **200** 独立 snapshot（每 tilt bin 等权） |
| obs/gt | 同 O0E0：`obs.npz` 无 \(p,R,n\)；estimator 签名不变 |

### 预注册 pose 分布（生成器保证 excitation，不事后挑帧）

\[
\beta\in\{0^\circ,10^\circ,20^\circ,35^\circ,50^\circ\}
\quad\text{等权},\qquad
\alpha\sim U[0,2\pi),\qquad
\gamma\sim U[0,2\pi)
\]

杯轴（world，桌面上为 \(e_z\)）：

\[
n=
\bigl(\sin\beta\cos\alpha,\;\sin\beta\sin\alpha,\;\cos\beta\bigr),\qquad
R=R_0(n)\,R_y(\gamma).
\]

等权 5 bin ⇒ \(P(\beta>15^\circ)=0.6\gg 0.30\)。  
\(\gamma\) 随机化：B2 理论上应对 yaw nuisance 不变（本格不测 B2，只保证 nuisance 存在）。

**位置**：桌面附近小盒子 \(p_{xy}\) + 固定略抬高 \(p_z\)（避免大倾角埋进桌面）。**不是** task-realistic placement。

## Gates

### G0a — observation support（阈值继承，不改）

\[
\boxed{P(V^{\mathrm{any}})\ge 0.98}
\]

另报（非门）：\(P(V^h),P(V^o),P(V^h\lor V^o)\)；失败帧 FOV / mask area。

### G0b — axis excitation（阈值继承）

\(n_{\mathrm{const}}=\) train GT 均值（归一化）。test：

\[
\boxed{r_{\mathrm{excite}}=P\bigl(\angle(n_t,n_{\mathrm{const}})>15^\circ\bigr)\ge 0.30}
\]

由 sampling 保证，不挑帧。生成器若没产出预期 tilt 分布 → 本门 FAIL。

### G0c — yaw nuisance coverage（instrument，非 axis science）

test 上每个 \(\beta\) bin：\(\gamma\) 落入 8 个 \(45^\circ\) octant 中 **≥6** 个非空。  
确保后续 B2 面对 \(SO(2)\) nuisance，而不是 yaw 被冻结。

## Patterns

| Pattern | 条件 |
|--------|------|
| `observation_support_failure` | ¬G0a |
| `axis_excitation_failure` | G0a ∧ ¬G0b |
| `yaw_nuisance_coverage_failure` | G0a ∧ G0b ∧ ¬G0c |
| `controlled_pose_observation_qualified` | G0a ∧ G0b ∧ G0c |

G0a 再 FAIL：**仍不加相机**；另开 camera-contract cell 才允许第三视角。

## 解锁

| Pattern | 解锁 |
|---------|------|
| `controlled_pose_observation_qualified` | 允许在 **本格 cache** 上重开 O0E0 **science**（冻结 B0/B1/B2；不改搜索） |
| 其余 | B2 仍 UNTESTED；O0E1 / O1 LOCKED |

Science 重开后的 B0 具科学意义：controlled 上 B0 FAIL + B2 PASS 才表明从观测恢复 axis，而非 task prior。**本格不写该 claim。**
