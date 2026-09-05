# RTWX-X0EH2 预注册 — Cross-Task External Validity

日期：2026-08-28  
状态：**已冻结（采集/训练/评分前）**  
依赖：X0EH1 = `receding_structure_capacity_shift`（\(R_P^{K=4}=0.988\) on `put_object_cabinet`）。  
**禁止**：改 M2 basis；改 \(H_{\mathrm{plan}}/K_{\mathrm{execute}}\)；改 matched/competent 阈值；扩 NN width grid 追更大 compression；改 X0E–X0EH1 ledger；TASK-XL/RGB/R10。

## 科学问题

\[
\boxed{
R_P^{K=4}>0
\text{ 是否在冻结结构与部署合同下跨 manipulation 动力学分布保持？}
\]

不是重新发明 structure；是 **外部有效性**。若新任务必须全新 basis，则 X0EH1 的 84× 更像单任务 inductive bias。

## 冻结合同（与 X0EH1 完全相同）

| 项 | 冻结值 |
|----|--------|
| 结构 | X0E-M2 basis \(\phi\)，\(P_{\mathrm{struct}}=1752\) |
| 部署 | \(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)，周期真实观测 reset |
| B0 | Pure MLP，\(H\in\{8,16,32,64,128,256\}\)，同 X0EH1 架构/训练协议 |
| B1 | M2 LS on **each task's train**（basis 不变，仅重拟合 \(W\)） |
| Reference | H=256 |
| Competent | \(r_{\mathrm{cat}}^{K=4}=0\)，\(E_{\mathrm{exec}}^{K=4}\le0.75\,E_{\mathrm{identity}}^{K=4}\) |
| Matched | \(E_{\mathrm{exec}}\le1.05\,E_{\mathrm{exec}}^{\mathrm{ref}}\)，\(E_{\mathrm{end}}\le1.10\,E_{\mathrm{end}}^{\mathrm{ref}}\)，\(r_{\mathrm{cat}}=0\) |
| \(R_P^{K=4}\) | \(1-P_{\mathrm{struct}}/P_{\mathrm{NN}}^{\min}\)（仅当该任务 reference competent 且 B1 matched） |

**不得**用 source task（`put_object_cabinet`）数据作 confirmatory test。

## 任务（预注册，评分前冻结）

| 角色 | RoboTwin task | seed family | split |
|------|---------------|-------------|-------|
| **Source（已报）** | `put_object_cabinet` | 12601 | X0EH1 已用 |
| **Holdout A** | `place_empty_cup` | **13601** | 48/24/24 × 120 |
| **Holdout B** | `stamp_seal` | **13602** | 48/24/24 × 120 |

两 holdout 均为 native-qpos、自由空间臂关节激励（同 X0E 采集协议）；与 source 不同的 manipulation 动力学分布。

可选第三 holdout（**仅 exploratory，不进 primary pattern**）：`adjust_bottle`，seed 13603。

## 每任务流程

1. 采集 train/val/test（fully fresh per task）
2. 各任务独立：train B0（5 seeds × width grid）、LS fit B1 \(W\)
3. Test 上 \(K=4\) receding 评估 B0/B1
4. 报告 \(E_{\mathrm{exec}}^{K=4}\)，\(E_{\mathrm{end}}^{K=4}\)，\(r_{\mathrm{cat}}^{K=4}\)，\(P_{\mathrm{NN}}^{\min}\)，\(R_P^{K=4}\)（若定义）
5. 报告 \(C_{\mathrm{deploy}}\)（同 X0EH1 proxy）

## Primary gates（per holdout task）

- **G_ref**：H=256 reference competent on test
- **G_struct**：B1 matched vs that task's H=256 ref
- **G_rp**：\(R_P^{K=4}>0\)（即 \(P_{\mathrm{NN}}^{\min}>1752\) 且 B1 matched）

## Patterns（primary，仅 A+B）

| Pattern | 条件 |
|---------|------|
| `cross_task_structure_supported` | A、B 均 G_ref、G_struct、G_rp |
| `partial_cross_task_transfer` | 恰一 holdout 满足三者 |
| `single_task_inductive_bias` | 两 holdout 均 G_ref 但 B1 均不 matched，或均 reference_failure |
| `reference_failure_on_transfer` | 任一 holdout reference 不 competent（该任务 \(R_P^{K=4}\)=undefined） |

若两 holdout reference 均 competent 且 B1 均 matched 但 \(R_P^{K=4}\) 量级差异 >10×，仍报 `cross_task_structure_supported`，但须在报告中分任务披露数值（禁止合并成更漂亮的 pooled ratio）。

## 明确不做

- 不 per-task 改 \(\phi\) 或加 residual
- 不 pooled train 跨任务（每任务独立 fit）
- 不回头优化 `put_object_cabinet` 上的 84×
- 不开 TASK-XL / RGB / R10

## 成功读数

`cross_task_structure_supported` ⇒ 支持 **cross-task structural compression** 假说，优于继续打磨单任务 compression 或过早开视觉栈。
