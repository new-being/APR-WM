# RTWX-O0E0R7 预注册 — Persistent Surface Reference (Static Multi-View)

日期：2026-08-31  
状态：**已冻结 / PAUSED（beam frontier 候选，非 auto-NEXT）**  
依赖：O0E0R6=`joint_inference_insufficient`；\(S_0\) / visibility LUT / B2 **冻结**  
**唯一新增机制**：跨时间 persistent surface \(M_K^{surf}\)（**非** O1 dynamics / **非** learned memory）

**禁止**：object dynamics；moving tracking；camera 重标定；uncertainty routing；gauge 重开；task attention；\(D_{ho}\) score 加权；改 B2；训练网络；R4 式等权 cloud union 作为 P1/P2。

## 科学问题

\[
\boxed{
\text{历史多视角能否建立稳定的 object-centric surface reference，
把 view-dependent center bias 压到 }{\sim}1\text{–}2\text{ cm？}
}
\]

R6 结论：单帧 decoupling 可消灭 catastrophic tail（P90 129°→33°），但无法提供稳定最终 center（\(n^{(2)}\) degradation）。R7 增加**跨时间真实 evidence**，reference 不随当前 axis candidate 漂移。

对象静止：\(T_{BO,t}=T_{BO}\)。已知 \(T_{BC,t}\)。

## 冻结组件

| 组件 | 状态 |
|------|------|
| \(S_0\) natural UNet | 冻结 |
| `visibility_lut.npz`（R5） | 冻结 |
| B2 search / objective | 冻结 |
| Controlled pose / 128² / dual camera | 冻结 |

## 数据

| 项 | 值 |
|----|-----|
| Multiview obs | **复用** R4 `mv_test` cache；seed **37604**；每 instance **8** frozen robot views |
| Segmentation | \(S_0\) on per-view RGB-D |
| \(K\) sweep | \(\{1,2,4,8\}\)（deterministic view order，与 R4 一致） |
| 新 formal single-view seed | **不** 用于本格 primary（multiview-only） |

每 view：\(S_0\) mask → \(\mathcal P_t^C\) → \(T_{BC,t}\) → \(\mathcal P_t^B\)。

## Persistent surface 三支（冻结）

### P0 — Naive union（R4 复现 baseline）

\[
M_K^{union}=\bigcup_{t=1}^{K}\mathcal P_t^B
\]

等权 per-view subsample + cap（R4 规则）。预期仍 ~20° 级 FAIL。

### P1 — Persistent voxel mean

Voxel grid：\(v(x)=\lfloor x/h\rfloor\)，**冻结** \(h=5\) mm。

每 voxel 记录 \((\bar x_j, w_j, c_j)\)：

- **Per-view reduction**：每个 view 对每个 voxel **最多贡献一次**（该 view 内点 median 或 single representative）
- 跨 view 更新：\(\bar x_j^{new}=(w_j\bar x_j+x_t)/(w_j+1)\)；\(c_j\) = 独立 view 支持计数（非 raw point count）

### P2 — Persistent consensus surface

P1 基础上仅保留：

\[
c_j\ge c_{\min}
\]

**冻结** \(c_{\min}=2\)（至少 2 个独立 view 支持）。

## Center 估计（冻结）

### Instrument（oracle axis \(n^{GT}\)）

从 \(M_K^{surf}\)（P1 或 P2 点集）：

\[
\boxed{
\hat p_K
=
\arg\min_p
\frac1{|M|}\sum_{x\in M}\min_{y\in\mathcal G_{CAD}(n^{GT})}\|x-(y+p)\|^2
}
\]

（CAD 点云用 GT axis 对齐的 021\_cup profile；**不** joint 优化 \(n\)）。

Diagnostic target：\(e_p(K)=\|\hat p_K-p^{GT}\|\)；报告 R4 机制尺度 **1.75 cm**（**非 hard gate**）。  
原 position gate不变：median \(\le5\) cm，\(E_p\le0.20\)。

### Formal

\[
RGBD_{1:K}\rightarrow M_K^{surf}\rightarrow \hat p_K\rightarrow \hat n_K
\]

\(\hat n_K=\arg\min_n S(n;\hat p_K)\)（冻结 B2）；**无** GT \(p,n\)。

## 三层门

### L1 — Data

同 P0 generator：\(P_{\mathrm{any}}\ge0.98\)；\(r_{\mathrm{excite}}\ge0.30\)；8/8 yaw octants。FAIL → STOP。

### L2 — Instrument（per \(K\)，P1/P2）

**G\_I1 position**：\(\hat p_K(n^{GT})\) 过原 position 门。  
**G\_I2 axis ceiling**：固定 \(\hat p_K\)，B2 → median axis \(\le15^\circ\)，P90 \(\le30^\circ\)。

Primary instrument：**P2**；P1 为 ablation；P0 为 sanity。

### L3 — Formal non-oracle（仅 L2 PASS 后）

门与 O0E0 相同（axis + position）。Per \(K\) 报告；**unlock 需某 \(K\) formal 双门 PASS**。

## \(K\) 稳定性（descriptive，非 post-hoc gate）

**全部** \(K\in\{1,2,4,8\}\) 报告 \(e_p(K)\)、\(e_{\mathrm{axis}}(K)\)。

Descriptive tags（非 unlock pattern）：

- `K_monotone_improving` — \(e_p\) 或 axis median 随 \(K\) 单调或 plateau（\(K=4\approx K=8\)）
- `K_unstable` — 某 \(K>1\) 显著劣于 \(K=1\)

**禁止** cherry-pick 最优 \(K\) 作为唯一 claim。

## Leave-one-view-out diagnostic

对每个 \(K\) 和 branch P1/P2：

\[
\Delta p_{-v}=\|\hat p_K-\hat p_K^{(-v)}\|
\]

报告 median / P90。与 P0 union 敏感性对照（descriptive）。

## Patterns

| Pattern | 条件 |
|--------|------|
| `persistent_surface_supported` | L3 formal（P2 或 P1）某 \(K\) axis+position PASS |
| `persistent_surface_formal_failure` | L2 PASS，L3 FAIL |
| `persistent_surface_instrument_failure` | L2 FAIL（GT axis + persistent center 不足） |
| `observation_support_failure` | L1 FAIL |

O0E1 **LOCKED** 除非 `persistent_surface_supported`（仅 prereg unlock，**不** 开 O1）。

## Failure chain（截至 R6）

```text
data ✓ → segmentation ✓ → cloud ✓ → B2 ✓
→ visibility center (oracle axis) ✓
→ finite joint inference ✗
→ persistent reference ← 本格
```

## 实现备忘

```text
aprwm_v0/geometry/persistent_surface.py   # P0/P1/P2 + LOO
aprwm_v0/rtwx_o0e0r7.py
runs/rtwx_o0e0r7/
```

## 后续（本格不跑）

```text
R7: persistent surface  [本格]
→ visibility-weighted fusion / CAD reg / learned surface（仅 R7 instrument FAIL）
→ O0E1 multi-symmetry（仅 persistent_surface_supported）
→ O1 persistent belief（仍 LOCKED）
```
