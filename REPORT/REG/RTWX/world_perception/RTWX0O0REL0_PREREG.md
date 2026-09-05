# RTWX-O0REL0 预注册 — Relative Effective-Pose Tracking Probe (2×2 Factorial)

日期：2026-08-31  
状态：**已冻结 / RAN**  
搜索策略：**beam #1**（单帧 absolute reference 族 **PAUSED**，\(D_{\max}=2\)）  
依赖：O0E0R6=`joint_inference_insufficient`；\(S_0\) dual-view cloud **冻结**  
**禁止**：visibility LUT；\(\hat p_{\mathrm{ref}}\)；B2；CAD canonical；解锁 O0E1/O1

## 唯一假设

\[
\boxed{
H_{\mathrm{rel}}:
\text{已知 }T_{BC}\text{ 时，两帧 partial RGB-D cloud 的相对刚体运动
比逐帧 absolute }(p,n)\text{ 更易闭合。}
}
\]

Estimator：\(\hat\Delta T_{01}=(\hat\Delta R,\hat\Delta t)\)。  
Formal 只评 effective state：\(\Delta p,\Delta n\)（**不**评 canonical yaw）。

## 2×2 Factorial（paired motion）

每 pair index \(j\) **共享**同一组：

- initial object pose \((p_0,R_0)\)
- object motion \(\Delta T_O^j\)
- camera-rig motion \(G_C^j\)

三 regime 仅改变“谁动”：

| Regime | Camera | Object | 回答问题 |
|--------|--------|--------|----------|
| **C0** | move | static | ego-motion cancellation |
| **C1** | static | move | pure relative tracking |
| **C2** | move | move | mixed real case |

C1↔C2 object motion **paired**；C0↔C2 camera motion **paired**。

| 项 | 值 |
|----|-----|
| Fresh seed | **37606** |
| 每 regime | **N=200** pairs |
| 总计 | **600** pairs；无训练 |

## Initial object pose

\(\beta_0\in\{0°,10°,20°,35°,50°\}\) 等权；\(\gamma_0\) 覆盖 **8 yaw octants**（P0 generator contract）。

## Object relative motion（冻结）

### Effective tilt

\[
\Delta\beta\in\{10°,20°,35°\}
\]

等权（**非** \(5,10,20°\)——避免 identity baseline 过原 15° 门）。

对 \(n_0=R_0 e_y\)，随机 \(u\perp n_0\)：

\[
R_{\mathrm{tilt}}=\exp([u]_\times\Delta\beta),\quad n_1=R_{\mathrm{tilt}}n_0
\]

独立 nuisance yaw：\(R_{\mathrm{yaw}}=\exp([n_1]_\times\Delta\gamma)\)，\(\Delta\gamma\) 均匀 8 octants。

\[
R_1=R_{\mathrm{yaw}}R_{\mathrm{tilt}}R_0
\]

### Translation

\[
\|\Delta p\|\in\{0,3,6\}\text{ cm}
\]

等权；方向 **deterministic sphere/grid catalog**（formal 后不可调）。

## Relative-motion excitation gate（L1b）

Identity baseline \(B_{\mathrm{ID}}:\hat\Delta T=I\)。在 **C1+C2** 上评价。

要求 **不能** 过 axis 正式门：

\[
\boxed{
\mathrm{median}\,e_{\Delta n}^{\mathrm{ID}}>15°
\quad\text{或}\quad
P90>30°
}
\]

（\(\Delta\beta\in\{10,20,35\}\) 下 identity median \(\sim20°\) 应自然 FAIL。）

FAIL → `relative_motion_excitation_failure` → STOP science。

## Camera rig motion（冻结）

Rigid rig \(G_C\in SE(3)\)：

\[
T_{BC,h,1}=G_C T_{BC,h,0},\quad T_{BC,o,1}=G_C T_{BC,o,0}
\]

\[
\|\Delta p_C\|\in\{2,4\}\text{ cm},\quad
\Delta\theta_C\in\{5°,10°\}
\]

axis/direction：**deterministic catalog**。

## Cache（物理隔离）

```text
runs/rtwx_o0rel0/cache/{c0,c1,c2}/
    obs.npz    # 无 GT
    gt.npz     # evaluator only
    seg_gt.npz # S0 train seal（若有）
```

`obs.npz` 仅：

```text
rgb0_h/o, depth0_h/o, K0_h/o, T_BC0_h/o
rgb1_h/o, depth1_h/o, K1_h/o, T_BC1_h/o
```

**禁止**：`p_gt`, `R_gt`, `n_gt`, `delta_T_gt` 进入 obs。

## Observation front-end（冻结）

\[
RGBD_t^{h,o}\xrightarrow{S_0}M_t\xrightarrow{d,K^{-1},T_{BC,t}}\mathcal P_{t}^{B}
\]

\[
\mathcal P_t^B=\mathrm{Fuse}(\mathcal P_{h,t}^B,\mathcal P_{o,t}^B)
\]

（dual-view fuse contract 与 O0E0 一致；**无** LUT/B2/CAD。）

### Voxel preprocess

\(h_v=0.02 D_O\)：per-view voxel centroid → concat → 同尺度再 voxel。**不** random subsample。

## Relative estimator（冻结 API）

`geometry/relative_rigid_registration.py`：

```python
@dataclass(frozen=True)
class RelativeICPConfig:
    voxel_frac: float = 0.02
    corr_frac: float = 0.30      # gate = corr_frac * D_O
    n_iters: int = 10
    min_corr: int = 16

def estimate_relative_rigid(cloud0_B, cloud1_B, object_diameter, cfg) -> RelativeRigidResult
```

- Init：\(R^{(0)}=I\)，\(t^{(0)}=c_1-c_0\)（observed cloud centroids only）
- Mutual NN；accept if \(\|X_i-Y_j\|\le 0.30 D_O\)
- Point-to-point Kabsch；\(T^{(k+1)}=\Delta T_k T^{(k)}\)
- \(N_{\mathrm{corr}}<16\) → `valid=false`（**保留** pair，evaluator 记 FAIL）

**禁止**：adaptive threshold；multiscale；colored ICP；RANSAC；early stop tuning。

## GT（evaluator only）

\[
\Delta T_O^{\mathrm{GT}}=T_{BO,1}T_{BO,0}^{-1}
\]

### Effective axis

\[
\hat n_1=\hat\Delta R\,n_0^{\mathrm{GT}},\quad
e_{\Delta n}=\arccos(\mathrm{clip}(\hat n_1^\top n_1^{\mathrm{GT}},-1,1))
\]

（\(n_0^{\mathrm{GT}}\) 仅 evaluator 用于检验 relative propagation。）

### Relative position

\[
\hat p_1=\hat\Delta R\,p_0^{\mathrm{GT}}+\hat t,\quad
e_{\Delta p}=\|\hat p_1-p_1^{\mathrm{GT}}\|
\]

**Claim scope**：relative transform capability **conditional on correct previous anchor**；非 absolute localization。

## 三层门

### L1a — Observation

\(P(V_0^{any}\land V_1^{any})\ge0.98\)；\(P(\hat M_0\neq\emptyset\land\hat M_1\neq\emptyset\mid V)\ge0.95\)。

FAIL → `relative_tracking_observation_failure` → STOP。

### L1b — Excitation

见上 → `relative_motion_excitation_failure`。

### L2 — C0 ego-motion instrument（严于 science）

\[
\mathrm{median}\,e_{\Delta n}\le5°,\quad P90\le10°
\]
\[
\mathrm{median}\,e_{\Delta p}\le2\text{ cm},\quad P90\le4\text{ cm}
\]

FAIL → `ego_motion_compensation_failure` → STOP C1/C2。

### L3 — C1 / C2 science（O0 尺度）

Axis：median \(\le15°\)，P90 \(\le30°\)。  
Position：**复用 O0E0 exact evaluator/gate**（median \(\le5\) cm，\(E_{\Delta p}\le0.20\)）。

**C1 与 C2 均须 PASS** → `relative_tracking_supported`。

## Paired absolute baseline（descriptive）

Frame-1 冻结 **R6 A1** \(n^{(1)}\)（\(n_{\mathrm{const}}\to p^{(0)}\to n^{(1)}\)）。  
**不参与** pattern rescue。

## Diagnostics（非 gate）

| ID | 内容 |
|----|------|
| D_gtmask | GT mask → 同 ICP；tag `relative_tracking_segmentation_coupled` vs `relative_geometry_insufficient` |
| D_overlap | \(\rho_{\mathrm{overlap}}\)：GT warp \(X_0'\) 与 \(\mathcal P_1\) 在 \(0.05D_O\) 内比例 |
| D_bins | 按 \(\Delta\beta\in\{10,20,35\}°\)；\(\|\Delta p\|\in\{0,3,6\}\) cm |
| D_cam | \(\Delta e_{\mathrm{cam}}^{(i)}=e_{C2}^{(i)}-e_{C1}^{(i)}\)（paired） |

## Patterns

```text
relative_tracking_observation_failure
relative_motion_excitation_failure
ego_motion_compensation_failure
relative_tracking_insufficient
relative_tracking_supported
```

```python
supported = data_pass and excitation_pass and c0_pass and c1_pass and c2_pass
```

Unlock：`unlocks_relative_tracking_branch=true`；**`unlocks_o0e1=false`**；**`unlocks_o1=false`**。

## FAIL 解读（不继续深搜 ICP）

| 结果 | 含义 |
|------|------|
| C0 FAIL | relative branch 优先级 ↓ |
| C0 PASS, C1 FAIL | partial-cloud rigid reg 不足；不调 20 个 ICP 参数 |
| C1 PASS, C2 FAIL | tag `camera_view_change_coupled` |
| C1+C2 PASS | relative branch → beam #1 |

## 实现

```text
aprwm_v0/rtwx_o0rel0.py
aprwm_v0/geometry/relative_rigid_registration.py
runs/rtwx_o0rel0/{summary,run,header}.json
runs/rtwx_o0rel0/per_pair.json
```

CLI：`rtwx-o0rel0 --output runs/rtwx_o0rel0`（**不**暴露 `corr_frac`/`voxel_frac`/`iters`/`motion-range`）。

## 单元测试（冻结清单）

1. Known rigid synthetic recovery  
2. Identity \(P_1=P_0\)  
3. Centroid init for pure translation  
4. \(\det R=+1\) reflection rejection  
5. Mutual-NN only  
6. Fixed 10 iterations  
7. Camera cancellation (static object, moved cam)  
8. Yaw gauge invariance of \(e_{\Delta n}\)  
9. No-GT estimator API  
10. S0/config hash frozen  
11. Identity baseline fails excitation on synthetic motion dist  

## 一句话

\[
\boxed{
\text{同一套 RGB-D observation 下，
absolute state recovery 改写为 relative rigid-motion recovery，
effective state 是否显著变简单？}
\]

PASS → 证据支持：可靠初始化 + 高频 relative tracking + 低频 absolute correction。
