# RTWX-O0E0R3 预注册 — S0 Formal Effective-Pose Confirmation

日期：2026-08-31  
状态：**已冻结 / RAN** `reference_coupled_failure`  
依赖：O0E0R2=`segmentation_cloud_failure`（\(S_0\) instrument PASS；\(S_1\) 放弃）；O0E0R1 quotient viable；B2 **冻结**  
**禁止**：修 \(S_1\)；改 B2；用 R2 fresh seed **37602** 作 formal claim；开 O0E1 除非 `effective_pose_supported`。

## 科学问题

\[
\boxed{
S_0\text{ natural-trained segmentation 在 fresh controlled test 上，
能否支撑 formal non-oracle }(p,n)\text{ closure？}
}
\]

\[
\boxed{\text{S0 cloud 闭合} \neq \text{formal effective pose 闭合}}
\]

R1：GT mask + reference → 20.3°；GT mask + GT \(p\) → 0.17°。本格检验 reference coupling 是否为 formal blocker。

## 冻结合同

| 项 | 值 |
|----|-----|
| Segmentation | **\(S_0\)** only：natural O0E0 cache train/val；O0G2 UNet recipe **不变** |
| B2 | R0/R1 完全冻结 |
| Formal test | **新** fresh controlled；seed **37603**；n=**200**；**≠37602**（R2） |
| Generator | 与 P0 相同 frozen `set_pose` contract |
| \(n_{\mathrm{const}}\) | P0 train GT（B0 only） |

## 三层门（顺序 STOP）

### L1 — Data（fresh test）

\[
P(V^{\mathrm{any}})\ge0.98,\quad r_{\mathrm{excite}}\ge0.30,\quad \text{yaw octants}\ge6/8\ \text{per }\beta
\]

FAIL → `observation_support_failure`；STOP。

### L2 — Instrument（\(S_0\)，oracle substitution）

\[
G_{I1}:\ \hat M^{S_0}+p^{GT}\xrightarrow{\text{B2}}\quad \mathrm{median}\le15^\circ,\ P90\le30^\circ
\]

\[
G_{I2}:\ \hat M^{S_0}+n^{GT}\ \Rightarrow\ \hat p\quad E_p\le0.20,\ \mathrm{median}\le5\text{ cm}
\]

FAIL → `segmentation_cloud_failure`；STOP science。

### L3 — Formal（non-oracle）

\[
\hat M^{S_0}+\hat p^{\mathrm{surf}}-\delta_y\hat n+\text{frozen B2}\rightarrow(\hat p,\hat n)
\]

同尺度 axis + position 门。

## Patterns（仅三格 + L1）

| Pattern | 条件 |
|--------|------|
| `observation_support_failure` | ¬L1 |
| `segmentation_cloud_failure` | L1 ∧ ¬L2 |
| `reference_coupled_failure` | L2 ∧ ¬L3 formal |
| `effective_pose_supported` | L3 formal PASS → unlock O0E1 prereg |

claim_scope：`controlled_021_cup_rgbd_S0_only`。
