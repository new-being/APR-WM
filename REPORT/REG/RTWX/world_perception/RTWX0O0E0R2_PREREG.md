# RTWX-O0E0R2 预注册 — Controlled-Pose Segmentation Repair + Contingent Science

日期：2026-08-31  
状态：**已冻结 / RAN** `segmentation_cloud_failure`  
依赖：O0E0R1=`quotient_viable_under_oracle_cloud`；O0E0R0=`effective_axis_failure`（**不变**）  
**禁止**：改 B2 search/objective；改 UNet architecture/loss/threshold/resolution；改 fusion；用 P0 test 作正式 test；开 O0E1 / O1。

## 核心假设

\[
\boxed{
H_{\mathrm{seg}}:
\text{R0 cloud corruption 主要来自 segmentation 的 pose-distribution mismatch}
}
\]

**唯一修改**：segmentation 训练分布

\[
\text{natural/upright (}S_0\text{)}
\quad\rightarrow\quad
\text{controlled multi-tilt P0 train/val (}S_1\text{)}.
\]

## 数据合同

| 项 | 值 |
|----|-----|
| \(S_0\) | 同 O0G2 UNet recipe；**natural** `place_empty_cup` train/val（O0E0 cache 128²） |
| \(S_1\) | 同 architecture/recipe；**P0 train/val** GT mask 监督 |
| 正式 test | **fresh** controlled generator；seed **37602**；n=**200**；与 P0 **同冻结 generator** |
| P0 test | **禁止**作 \(S_1\) 正式 test（R0/R1 已查看） |
| B2 | 与 R0/R1 **完全冻结** |

## Instrument gates（\(S_1\) on fresh test）

### \(G_{I1}\) — axis-cloud closure

learned mask + **GT \(p\)** + frozen B2：

\[
\boxed{\mathrm{median}(e_{\mathrm{axis}})\le15^\circ,\quad P90\le30^\circ}
\]

\(S_0\) baseline 同期报告（R1 参照 **74.4°**）。

### \(G_{I2}\) — position-cloud closure

learned mask + **GT \(n\)**：

\[
\hat p=\hat p^{\mathrm{surf}}(\hat M)-\delta_y n^{GT}
\]

\[
\boxed{E_p\le0.20,\quad \mathrm{median}(e_p)\le5\ \mathrm{cm}}
\]

任一 FAIL → **`segmentation_cloud_failure`**；**STOP science**。

## Contingent formal science（预授权，非新格）

仅当 \(G_{I1}\land G_{I2}\) PASS：

\[
\text{learned mask}+\text{reference center}+\text{frozen B2}
\]

沿用 R0 正式门 + B0 exclusion（B0 不过 G1）。

## Diagnostics（非门）

1. mask IoU / precision / recall vs \(\beta\)  
2. head vs observer 分视图  
3. \(\rho_A=|\hat M|/|M^{GT}|\)；\(\Delta c_{\mathrm{mask}}\)  
4. cloud centroid displacement、point-count ratio、NN distance  

## Patterns

| Pattern | 条件 |
|--------|------|
| `segmentation_cloud_failure` | \(S_1\)：¬\(G_{I1}\) 或 ¬\(G_{I2}\) |
| `reference_coupled_failure` | \(G_I\) PASS；正式 axis/position FAIL |
| `effective_pose_supported` | \(G_I\) + 正式全 PASS + B0 excluded |
| `segmentation_training_insufficient` | mechanism tag：\(S_1\) 相对 \(S_0\) 无实质 instrument 改善 |

**不**改 R0/R1 pattern。不解锁 O0E1 / O1。
