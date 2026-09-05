# RTWX-O0E0R0 预注册 — Controlled-Pose Effective-Pose Science Recovery

日期：2026-08-31  
状态：**已冻结 / RAN** `effective_axis_failure`  
依赖：O0E0P0=`controlled_pose_observation_qualified`；O0E0 natural=`observation_support_failure`（B2 **UNTESTED**）  
**禁止**：重采数据；筛帧；改 tilt/yaw 分布；改 Fibonacci/B2 搜索；开 O0E1 / O1；把 GT 喂给 estimator；用 natural-task 外推 claim。

## 科学问题（唯一）

\[
\boxed{
\text{在 O0E0P0 qualified cache 上，冻结的 quotient-CAD estimator 能否恢复 }(p,n)?
}
\]

本格 **不问**：natural task 是否需要 axis；yaw 因果意义；temporal tracking；O1 memory；新网络学 axis。

## 数据合同

| 项 | 值 |
|----|-----|
| 输入 cache | **仅** O0E0P0 `cache/{train,val,test}/obs.npz` + `gt.npz` |
| 重采 | **禁止** |
| 筛帧 | **禁止** |
| B0/B1/B2 | 与 O0E0 完全冻结（Fibonacci 162；Top-8；\(5^\circ/3^\circ/1^\circ\)；2D KDTree；one-sided mean-min-sq） |
| obs/gt | 物理隔离不变；estimator 不得读 `p_gt/R_gt/n_gt` |
| UNet | O0G2 家族；在 P0 **train** GT mask 上训练；**非**新 segmentation 实验 |

## Gates

### G0 — learned detection（继承 O0E0）

P0 已 \(V^{any}=1\)，等价：

\[
\boxed{P(\hat M\neq\varnothing)\ge 0.95}
\]

失败 → `controlled_pose_detection_failure`；**STOP**；不评 B2。

GT mask 仅允许 **ceiling diagnostic** 分支，不得进入正式 estimator。

### G1 — effective axis（B2 primary）

\[
e_{\mathrm{axis}}=\arccos\bigl(\mathrm{clip}(\hat n^\top n,-1,1)\bigr)
\]

**不**取 \(|\hat n^\top n|\)（\(SO(2)_y\) 不 quotient axis flip）。

\[
\boxed{\mathrm{median}(e_{\mathrm{axis}})\le15^\circ},\qquad
\boxed{\mathrm{P90}(e_{\mathrm{axis}})\le30^\circ}
\]

另报 \(f_{90}=P(e_{\mathrm{axis}}\ge75^\circ)\)（非门）。

### G2 — position（B2）

\[
\hat p=\hat p^{\mathrm{surf}}-\delta_y\hat n
\]

\[
\boxed{E_p\le0.20},\qquad
\boxed{\mathrm{median}(e_p)\le5\ \mathrm{cm}}
\]

### B0 exclusion（解释性，非数值差分门）

P0 保证 \(r_{\mathrm{excite}}=0.6\)。主 claim 需：

\[
\boxed{\text{B2 过 G1/G2，且 B0 被 G1 绝对门排除（B0 FAIL G1）}}
\]

若 B0 与 B2 均过 G1：**不**写 `effective_pose_supported` → `constant_baseline_not_excluded`。

**不**设 “B2 必须比 B0 好 X%” 事后阈值。

## Mechanism diagnostics（只读，非门）

### A — 按 tilt \(\beta\)

\(\beta\in\{0,10,20,35,50\}^\circ\)：各报 B2 \(\mathrm{median}(e_{\mathrm{axis}})\)、\(\mathrm{P90}(e_{\mathrm{axis}})\)。

### B — 按 yaw octant

固定 \(\beta\)（默认 \(20^\circ\)）下 8 个 \(\gamma\) octant 的 B2 axis error。

### C — oracle-axis position ceiling

\[
\hat p^{\mathrm{ceil}}=\hat p^{\mathrm{surf}}-\delta_y n^{GT}
\]

- ceiling PASS、B2 position FAIL → axis error 传播  
- ceiling FAIL → surface/reference chain 问题  

GT **仅**进入此 diagnostic。

## Patterns

| Pattern | 条件 |
|--------|------|
| `controlled_pose_detection_failure` | ¬G0 |
| `effective_axis_failure` | G0 ∧ ¬G1 |
| `effective_position_failure` | G0 ∧ G1 ∧ ¬G2 |
| `constant_baseline_not_excluded` | G0 ∧ G1 ∧ G2 ∧ B0 过 G1 |
| `effective_pose_supported` | G0 ∧ G1 ∧ G2 ∧ B0 不过 G1 |

full-\(R\) error：**diagnostic only**，非门。

## 解锁

| Pattern | 解锁 |
|---------|------|
| `effective_pose_supported` | O0E1 prereg（claim 限定 controlled `021_cup` RGB-D） |
| 其余 | O0E1 / O1 **LOCKED** |

**不得**外推：natural `place_empty_cup` 需要逐帧估 \(n\)（natural \(r_{\mathrm{excite}}=0.126\) 为 task prior diagnostic）。
