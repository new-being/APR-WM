# RTWX-O0E0 预注册 — Symmetry-Aware Effective-Pose Closure

日期：2026-08-31  
状态：**已冻结 / RAN** `observation_support_failure`  
依赖：O0G6A=`global_geometry_ambiguous`；O0G2R position PASS；O0A0=`appearance_yaw_generalization_failure`；O0G1b \(\delta_O\)；SYM-X 表示合同（\(\bar s,\gamma\) gauge）  
**旁支**：O0Q* = SIDE / FROZEN（无 causal quotient 结论）。O0T0 = PREREGISTERED / **NOT RUN / DEPRIORITIZED**  
**禁止**：yaw classification / temporal yaw；canonical correspondence / CAD-anchor identity；RANSAC/ICP 找唯一 full \(R\)；GT mask / GT \(R\) / GT axis 进入 primary B2；拟合 \(\delta_O\)；改 O0G2R/O0C/O0A0 门；开 O1 / O0E1 除非本格 PASS。

## 战略转向（本格之前已冻结）

O0A0 已表明单帧 RGB-D **不**编码 canonical yaw（train 亦为 chance）。  
继续 `history→yaw` 仅在**任务明确需要 yaw**（把手、logo、方向摩擦等）时才有 IG。

本格不问「simulator full \(R\) 能否被视觉唯一恢复」，而问：

\[
\boxed{
\text{若对象几何对称类 }G_{\mathrm{geom}}\text{ 已知，}
\text{视觉能否可靠恢复姿态等价类 }(p,[R]_G)\text{，}
\text{而不预测不可辨的 gauge？}
}
\]

措辞为 **effective geometric pose**，**不**声称已从因果动力学证明 \(R\to n\) quotient（O0Q 未闭合）。

## 对象与对称先验

| 项 | 值 |
|----|------|
| 资产 | `021_cup`（无柄近轴对称杯） |
| \(G_{\mathrm{geom}}\) | **\(SO(2)_y\)**（绕杯轴旋转 gauge） |
| 有效 orientation | 杯轴 \(n=Re_y\in S^2\)，\([R]=\{RR_y(\theta)\}\) |
| 有效 pose target | \(\boxed{s_{\mathrm{pose}}^{\mathrm{eff}}=(p,n)}\) |
| \(\delta_O\) | `model_data0.json` `center*scale`（冻结；**不**重拟合） |
| 轴上 reference | \(\delta_y=\|\delta_O\|_y\)（object frame 沿 \(Y\)）；\(\hat p=\hat p^{\mathrm{surf}}-\delta_y\hat n\) |

## 科学问题

> 在 **fully non-oracle** 双视角 RGB-D 接口下，能否稳定恢复 \((\hat p,\hat n)\)，使 axis error 与 position error 同时过门？

\[
\boxed{
RGBD^{\mathrm{head,obs}}
\rightarrow
(\hat p,\hat n)
}
\]

**不测** canonical yaw。full-\(SO(3)\) error 仅作 diagnostic。

## 冻结合同

| 项 | 值 |
|----|------|
| \(\mathcal O_t\) | `head` + `observer`（O0V；不可换） |
| Segmentation | **learned** U-Net（O0G2 family；与 O0G2R 同线） |
| Point cloud | \(\mathcal P_B\) = 双视角 learned mask ∧ finite depth 并集 |
| CAD | `visual/base0.glb` × scale（同 O0G5/O0G6） |
| Axis primary | **几何** quotient CAD matching；**不用 NN 作 axis primary** |
| seeds | **36601**（formal）；**48/24/24 × 120** |
| RGB-D | **128²**（与 O0G5C/O0G6A 线对齐） |

### Branches

| ID | 定义 | 角色 |
|----|------|------|
| **B0** | 常数轴 \(\hat n=n_{\mathrm{mean}}^{\mathrm{train}}\)（+ 同 position pipeline） | 排除「杯子永远竖直」假 PASS |
| **B1** | 简单 whole-shape 轴：PCA / cylinder-axis / 主方向启发式 | 简单几何是否已够 |
| **B2** | \(\mathcal P_B\leftrightarrow\mathcal M_O/G\)：在 \(S^2\) 上搜 \(\hat n=\arg\min_n D_{\mathrm{obs}\to\mathrm{CAD}}(\mathcal P_B,\mathcal M(n))\)；yaw gauge 忽略 | **primary** |

Position（B0/B1/B2 共用 learned-mask fusion）：

\[
\hat p^{\mathrm{surf}}=\mathrm{median}(\mathcal P_B),\qquad
\hat p=\hat p^{\mathrm{surf}}-\delta_y\hat n.
\]

**禁止**用 GT \(R\) 做 reference correction。

### B2 目标与 \(S^2\) 搜索（实现前一次性冻结）

CAD yaw-invariant profile：\(c_j^O=(h_j,r_j)=(y_j,\sqrt{x_j^2+z_j^2})\)。

\[
S(n)=\frac1N\sum_i\min_j\|c_i(n)-c_j^O\|^2
\quad\text{（obs→CAD 单向；mean of min sq；非对称 Chamfer）}
\]

| 项 | 冻结值 |
|----|--------|
| CAD 采样 | 2048 pts（FPS seed **36611**） |
| obs 子采样 | ≤256 / frame |
| coarse \(S^2\) | Fibonacci **K=162**（seed **36612**） |
| Top-\(K_r\) | **8** |
| local refine | tangent steps **5° / 3° / 1°**（各 8 邻域） |
| KDTree | 2D on CAD \((h,r)\) |

obs/gt **物理分文件**：estimator 签名不得接收 `p_gt/R_gt/n_gt`。

## Gates

### G0 — observation support

与 O0G2R/O0C 同尺度：

- agg \(P(V^{\mathrm{any}})\ge0.98\)；每 seed \(\ge0.95\)
- 联合 \(P(\hat M^{\mathrm{any}}\neq\emptyset\mid V^{\mathrm{any}})\ge0.95\)

失败 → `observation_support_failure`。**STOP**。

### G0b — axis excitation（fresh test GT）

防止 constant-up 假 PASS。在 **fresh test** 上（GT \(n_t\)）：

\[
P\bigl(\angle(n_t,n_{\mathrm{mean}})>\alpha\bigr)\ge\rho.
\]

冻结：**\(\alpha=15^\circ\)**，**\(\rho=0.30\)**。

失败 → `axis_excitation_failure`。**STOP**（需合法多姿态数据，不得用 constant-up 混过）。

### G1 — effective axis（B2 primary）

\[
e_{\mathrm{axis}}=\arccos\bigl(\mathrm{clip}(\hat n^\top n,-1,1)\bigr)
\]

**不**取 \(|\hat n^\top n|\)：\(SO(2)_y\) 只 quotient 绕轴旋转，**不** quotient axis flip（杯口上/下不同）。

沿用 orientation 主线尺度：

\[
\boxed{\mathrm{median}(e_{\mathrm{axis}})\le15^\circ},\qquad
\boxed{\mathrm{P90}(e_{\mathrm{axis}})\le30^\circ}.
\]

另报 \(f_{90}=P(e_{\mathrm{axis}}\ge75^\circ)\)。  
机制期望：full-\(R\) 的 90° 重尾在 \(e_{\mathrm{axis}}\) 上应基本消失。

### G2 — effective position（B2 primary）

沿用 O0G2R / O0C：

\[
\boxed{E_p\le0.20},\qquad
\boxed{\mathrm{median}\|e_p\|\le5\,\mathrm{cm}}
\]

（另报 strong \(\le2\) cm）。  
核心：用 \(\hat n\) 替代 GT/full \(R\) 后，reference-offset correction 是否仍厘米级。

### G3 — above constant axis

\[
e_{\mathrm{axis}}^{B2} < e_{\mathrm{axis}}^{B0}
\]

（或 axis \(E\) 等价改进；防止 B0 常数竖直轴假 PASS）。

### Diagnostic（非门）

对任意 representative \(\hat R\)（如由 \(\hat n\) + 任意 yaw gauge 构造）：

\[
d_{SO(3)}(\hat R,R)\quad\text{仅 diagnostic，不作 gate。}
\]

## Patterns（互斥）

| Pattern | 条件 |
|---------|------|
| `observation_support_failure` | ¬G0 |
| `axis_excitation_failure` | G0 ∧ ¬G0b |
| `effective_axis_failure` | G0∧G0b ∧ ¬G1 |
| `effective_position_failure` | G1 ∧ ¬G2 |
| `effective_pose_supported` | G0∧G0b∧G1∧G2∧G3 |

## 解锁

| Pattern | 解锁 |
|---------|------|
| `effective_pose_supported` | 允许预注册并实现 **O0E1**（multi-symmetry object pose） |
| 其余 | O0E1 / O1 **LOCKED** |

**O0 simulator GT 仍为 full \((T,R)\)**；本格 closure claim 针对 **effective perception target** \((p,n)\)，不是改写 RoboTwin state vector 定义。

## 表示合同（与 SYM-X 对齐）

对当前杯：

\[
\bar s=(p,n_{\mathrm{cup}}),\qquad \gamma=\theta_{\mathrm{yaw}}\ \text{(dormant)}.
\]

O0E0 PASS 只说明 **几何对称类下有效 pose 可感知**；\(\gamma\) 激活仍须任务/因果证据（SYM-X2 线），本格不做。
