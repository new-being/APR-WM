# RTWX-X0EH2 报告 — Cross-Task External Validity

日期：2026-08-28  
状态：**正式冻结** `cross_task_structure_supported`  
预注册：`REPORT/REG/RTWX/RTWX0EH2_PREREG.md`  
产物：`runs/rtwx_x0eh2/{run.json,metrics.json,summary.json}`（JSON 自 `run.log` 恢复；进程 stdout 落盘时未写 per-task 子目录）  
部署合同：\(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)；与 X0EH1 完全相同。  
Source（仅对照，非本格 confirmatory）：`put_object_cabinet` seed 12601（X0EH1）。

## 一句话

在冻结 M2 basis（\(P=1752\)）与 \(K=4\) receding 合同下，两个 **holdout manipulation 任务** 均满足 G_ref、G_struct、G_rp：  
**\(R_P^{K=4}=0.988\)**，**\(C_P^{K=4}\approx84.0\times\)**（每任务 \(P_{\mathrm{NN}}^{\min}=147224\)，H=256）。  
X0EH1 的 84× 压缩 **不是** `put_object_cabinet` 单任务 inductive bias；结构容量在跨动力学分布上保持。

## Holdout 任务与数据

| 任务 | seed | split | n_dof |
|------|------|-------|------:|
| `place_empty_cup` | 13601 | 48/24/24 × 120 | 12 |
| `stamp_seal` | 13602 | 48/24/24 × 120 | 12 |

每任务独立采集、独立 LS 拟合 \(W\)、独立 B0 训练；basis \(\phi\) 未改。

## Primary gates（per holdout）

| 任务 | G_ref | G_struct | G_rp | Pattern 贡献 |
|------|:-----:|:--------:|:----:|:------------|
| `place_empty_cup` | ✓ | ✓ | ✓ | 全通过 |
| `stamp_seal` | ✓ | ✓ | ✓ | 全通过 |

**Primary pattern：`cross_task_structure_supported`**

两任务 \(R_P^{K=4}\) 量级相同（均为 0.988），无 >10× 分任务差异需额外披露。

## Test 主表 — `place_empty_cup`（720 segments / 24 ep）

Reference competent（H=256）：\(r_{\mathrm{cat}}=0\)，\(E_{\mathrm{exec}}=0.601\le0.75\times1.211\)（identity）。**通过。**

| 模型 | \(P\) | \(E_{\mathrm{exec}}\) | \(E_{\mathrm{end}}\) | \(E_{\mathrm{id}}\) | \(r_{\mathrm{cat}}\) | matched |
|------|------:|------------------------:|----------------------:|----------------------:|---------------------:|:-------:|
| **B1 M2** | **1752** | **0.566** | **0.522** | 1.211 | **0** | **✓** |
| B0 H=8 | 656 | 1.129 | 1.476 | 1.211 | 0 | ✗ |
| B0 H=16 | 1544 | 1.083 | 1.408 | 1.211 | 0 | ✗ |
| B0 H=32 | 4088 | 1.008 | 1.279 | 1.211 | 0 | ✗ |
| B0 H=64 | 12248 | 0.821 | 0.954 | 1.211 | 0 | ✗ |
| B0 H=128 | 40856 | 0.640 | 0.629 | 1.211 | 0 | ✗ |
| B0 H=256 (ref) | 147224 | 0.601 | 0.565 | 1.211 | 0 | ✓ |

B1 相对 ref：\(E_{\mathrm{exec}}\) 约 **6% 更低**，\(E_{\mathrm{end}}\) 约 **8% 更低**。

容量：\(R_P^{K=4}=0.988\)，\(C_P^{K=4}\approx84.0\times\)。

## Test 主表 — `stamp_seal`（720 segments / 24 ep）

Reference competent（H=256）：\(r_{\mathrm{cat}}=0\)，\(E_{\mathrm{exec}}=0.659\le0.75\times1.249\)。**通过。**

| 模型 | \(P\) | \(E_{\mathrm{exec}}\) | \(E_{\mathrm{end}}\) | \(E_{\mathrm{id}}\) | \(r_{\mathrm{cat}}\) | matched |
|------|------:|------------------------:|----------------------:|----------------------:|---------------------:|:-------:|
| **B1 M2** | **1752** | **0.662** | **0.621** | 1.249 | **0** | **✓** |
| B0 H=8 | 656 | 1.184 | 1.579 | 1.249 | 0 | ✗ |
| B0 H=16 | 1544 | 1.164 | 1.541 | 1.249 | 0 | ✗ |
| B0 H=32 | 4088 | 1.119 | 1.466 | 1.249 | 0 | ✗ |
| B0 H=64 | 12248 | 1.061 | 1.363 | 1.249 | 0 | ✗ |
| B0 H=128 | 40856 | 0.898 | 1.082 | 1.249 | 0 | ✗ |
| B0 H=256 (ref) | 147224 | 0.659 | 0.664 | 1.249 | 0 | ✓ |

B1 与 ref 几乎持平（\(E_{\mathrm{exec}}\) 在 matched 裕度内略高 0.5%），\(E_{\mathrm{end}}\) 约 **6% 更低**。

容量：\(R_P^{K=4}=0.988\)，\(C_P^{K=4}\approx84.0\times\)。

## 跨任务对照（含 source，source 非 confirmatory）

| 任务 | 角色 | B1 \(E_{\mathrm{exec}}\) | ref \(E_{\mathrm{exec}}\) | \(R_P^{K=4}\) | \(C_P^{K=4}\) |
|------|------|-------------------------:|--------------------------:|--------------:|--------------:|
| `put_object_cabinet` | source (X0EH1) | 0.615 | 0.737 | 0.988 | ≈84.0× |
| `place_empty_cup` | holdout A | 0.566 | 0.601 | 0.988 | ≈84.0× |
| `stamp_seal` | holdout B | 0.662 | 0.659 | 0.988 | ≈84.0× |

三任务上 **最小 matched NN 宽度均为 H=256**；H=128 在两 holdout 上均不 matched（与 X0EH1 一致）。

## 部署算力（\(H_{\mathrm{plan}}=16\) / \(K=4\) proxy）

| 任务 | B1 M2 (ms/action) | B0 H=256 (ms/action) | 比值 |
|------|------------------:|---------------------:|-----:|
| `place_empty_cup` | 0.040 | 3.18 | ≈80× |
| `stamp_seal` | 0.040 | 3.16 | ≈79× |

与 X0EH1（≈76×）同量级。

## Pattern 与门

| 检查 | `place_empty_cup` | `stamp_seal` |
|------|:-----------------:|:------------:|
| G_ref | ✓ | ✓ |
| G_struct | ✓ | ✓ |
| G_rp (\(P_{\mathrm{NN}}^{\min}>1752\)) | ✓ | ✓ |

Pattern：**`cross_task_structure_supported`**  
`cross_task_capacity_claim=true`

## 与机制链的关系

```text
X0EH1  → receding_structure_capacity_shift (source only)
X0EH2  → cross_task_structure_supported
         R_P^(K=4)=0.988 on both holdouts; C_P^(K=4)≈84× preserved
```

1. **外部有效性成立**：冻结 \(\phi\) + per-task \(W\) 足以在两个新 manipulation 分布上复现 X0EH1 级容量结论。
2. **不是单任务过拟合**：若仅为 cabinet inductive bias，holdout 应出现 `partial_cross_task_transfer` 或 `single_task_inductive_bias`；未观察到。
3. **`stamp_seal` 上 B1 与 H=256 几乎打平**，说明结构在该任务上接近 matched 下界，但仍在 gate 内且 \(r_{\mathrm{cat}}=0\)。
4. 仍 **不是** torque-level physics ID；仍是 native-window macro-transition capacity under \(K=4\) contract。

## 账本

见 `REPORT/REP/RTWX/RTWX_LEDGER.md`。

```text
RTWX-X0EH2 = PASS / cross_task_structure_supported
             holdouts: place_empty_cup (13601), stamp_seal (13602)
             R_P^(K=4)=0.988, C_P^(K=4)≈84.0× per task
NEXT       = (未预注册；禁止事后扩任务追 ratio)
TASK-XL / R10 = LOCKED
```

**禁止事后**：per-task 改 basis、扩 width、改 matched 阈值、合并三任务 pooled ratio 粉饰数字。
