# RTWX-O0G6A 预注册 — Whole-Object Geometry Orientation Observability

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G5C=`canonical_identity_generalization_failure`（刚体 cup 上 **停止** canonical-identity 分支）；O0G2R position PASS；O0C `orientation_failure`（90° 重尾）  
**禁止**：ICP / RANSAC / 全局搜索作 primary；再训 B1/B2 descriptor；改 O0G5* 门；learned correspondence；解锁 O1 / O0C2。

## 研究决策（本格之前已冻结）

对当前**刚体 cup**，不再要求视觉恢复局部 canonical identity。已否：

\[
\text{dense }x^O,\quad
\text{sparse keypoints visibility},\quad
\text{FPFH},\quad
\text{learned discrete anchors (train PASS / fresh FAIL).}
\]

这**不**声称 identity “视觉上根本不可观测”——只否掉当前表示；但对刚体 pose 主线，继续加 descriptor 不是最高 IG。

## 科学问题

> 已有双视角 segmented RGB-D 点云、**不知道任何 pixel 的 canonical identity** 时，只拿整个观测形状与 CAD 比，是否足以确定 \(R_{BO}\)？

\[
\boxed{
\mathcal P_B^{\mathrm{cup}}
\;\leftrightarrow\;
\mathcal M_O
}
\]

不学习 correspondence。不跑 ICP。本格只问：

\[
\text{几何目标函数有没有正确 orientation signal？}
\]

不是：

\[
\text{optimizer 能不能找到那个 minimum。}
\]

## 冻结合同

| 项 | 值 |
|----|------|
| 数据 | 复用 O0G5C **128²** cache（seeds **33601 / 33602 / 33603**；12×120；GT mask） |
| \(\mathcal O_t\) | `head` + `observer` |
| \(\mathcal P_B\) | 双视角 mask∧finite xyz **并集**（world/base；与 O0G2 同线） |
| CAD \(\mathcal M_O\) | `visual/base0.glb` × scale（同 O0G5B）；FPS **2048**（seed 34601） |
| Translation | **\(p^{GT}\)** 固定（nuisance）。不用 O0G2R \(\hat p\) 作 primary |
| Mask | **GT**（隔离 shape-vs-CAD；非 localization 格） |
| 子采样 | 每帧最多 **256** obs 点 |
| **不做** | ICP；RANSAC；Kabsch primary；训练 |

### 单向 metric（partial-shape）

\[
S(R)
=
\frac1{|P|}
\sum_{x\in P}
\min_{y\in R\mathcal M_O+p}
\|x-y\|^2
\]

不使用对称 Chamfer（会惩罚被遮挡的 CAD 面）。

### Hypothesis 集（相对 \(R^{GT}\)，运行前冻结）

\[
R_k^{\mathrm{hyp}}=R^{GT}R_k,\qquad k=1,\ldots,72
\]

生成器 \(R_k\)（seed **34601**）：

- \(I\)；
- 8 个 \(10^\circ\) 局部扰动（GT basin）；
- 轴角 \(\{x,y,z\}\times\{90^\circ,180^\circ,270^\circ\}\)；
- 其余：均匀 SO(3)（互距 \(\ge5^\circ\)）。

GT basin：

\[
\mathcal N=\{R_k:d(R_k,I)\le15^\circ\}.
\]

## Gates

| Gate | 条件 |
|------|------|
| G0-support | \(P(N_{\mathrm{usable}}\ge64)\ge0.90\)；每 seed \(\ge0.85\)；且 median \(\sqrt{S(I)}/D_O\le0.15\)（GT 拟合 sanity） |
| G1-top1 | \(P\big(d(R^*,R^{GT})\le15^\circ\big)\ge0.80\)；每 seed \(\ge0.70\) |
| G2-margin | median \(m>0\)，\(m=S(R_{\mathrm{wrong}}^{\mathrm{best}})-S(R_{\mathcal N}^{\mathrm{best}})\) |

Confirmatory（不改门）：Top-5 basin recall；\(P(e_{R^*}\in[75,105])\)；  
\(S(R^{GT})\) vs \(S(R^{GT}R_{90})\) 对 \(x,y,z\) 轴（杯轴≈object \(Y\)）。

## Patterns（互斥）

| Pattern | 条件 |
|--------|------|
| `surface_support_failure` | ¬G0 |
| `global_geometry_ambiguous` | G0 ∧ ¬(G1 ∧ G2) |
| `global_shape_orientation_supported` | G0 ∧ G1 ∧ G2 |

`global_geometry_ambiguous`：可见几何不足以稳定恢复 **full** \(R\)（含 90° cost 接近）。下一步才是 appearance / task-relevant quotient。

## 解锁

`global_shape_orientation_supported` → 允许预注册 **O0G6R**（coarse-to-fine / global registration / ICP **refinement**）。  
否则 O0G6R **LOCKED**。O0G5R / O0C2 / O1 **LOCKED**。
