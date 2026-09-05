# RTWX-O0G5A 预注册 — CAD Descriptor Observability（无训练）

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G4=`sparse_keypoints_unobservable`；O0G3B/A/R 已刻画 dense/sparse 失败  
**禁止**：训练 descriptor；RANSAC/KANSAC 作主结论；改 O0G3R 的 64²；扫 τ 救格；解锁 O1/O0C2。

## 科学问题

> 在 **新** 128² RGB-D 合同下，可见表面点能否通过 **冻结几何 descriptor（FPFH）** 检索到正确 CAD canonical 区域？

\[
j^*=\arg\min_j\|\phi_i^{obs}-\phi_j^{CAD}\|,\quad
e_{\mathrm{match}}^{norm}=\|\hat x_i^O-x_i^{O,GT}\|/D_O
\]

GT pose **仅用于评价**；matching 不使用 \(R^{GT},p^{GT}\)。

**不**跑 robust registration；solver ceiling 已由 O0G3/O0G4 闭合。

## 冻结合同

| 项 | 值 |
|----|------|
| 分辨率 | **128×128** RGB-D（新 interface；非救 O0G3R） |
| \(\mathcal O_t\) | head + observer |
| CAD | `021_cup/visual/base0.glb`；object frame 采样 **8192** 点 |
| Descriptor | **Open3D FPFH**（冻结；无学习） |
| seeds | **31601 / 31602 / 31603** |
| 规模 | 每 seed **12 test ep × 120** |
| 可用表面 | \(N_{\mathrm{usable}}\ge N_{\mathrm{desc}}=64\)（双视角 mask∧depth 并集） |
| 评价子采样 | 每帧最多 **256** obs 点（冻结 seed） |

## Gates

| Gate | 条件 |
|------|------|
| G0-support | agg \(P(N_{\mathrm{usable}}\ge64)\ge0.90\)；每 seed \(\ge0.85\) |
| G1-ambiguity | ambiguous-pair 比例 \(<0.15\)（descriptor 近、\(x^O\) 远） |
| G2-matching | med \(e_{\mathrm{match}}^{norm}\le0.10\)；P90 \(\le0.25\)；Top-5 recall \(\ge0.75\)（@ \(0.05D_O\)） |

Secondary（描述）：\(\lambda_2(C)\) of matched \(\hat x^O\) spread；per-view breakdown。

## Patterns（互斥）

| Pattern | 条件 |
|---------|------|
| `cad_surface_support_failure` | ¬G0 |
| `local_descriptor_ambiguous` | G0 ∧ ¬G1 |
| `cad_descriptor_observable` | G0 ∧ G1 ∧ G2 |

G0∧G1∧¬G2 → 仍记 `local_descriptor_ambiguous`（匹配不可靠）。

## 解锁

`cad_descriptor_observable` → 允许预注册 **O0G5R**（learned mask + descriptor matching + robust registration）。  
`local_descriptor_ambiguous` → learned/global-context descriptor 路线。  
`cad_surface_support_failure` → 审查 observation contract / 视角。  
**O0C2 / O1 LOCKED**。
