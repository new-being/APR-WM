# RTWX-O0E0R6 报告 — Finite-Step Alternating Center–Axis Inference

日期：2026-08-31  
状态：**正式冻结** `joint_inference_insufficient`  
预注册：`REPORT/REG/RTWX/world_perception/RTWX0O0E0R6_PREREG.md`  
产物：`runs/rtwx_o0e0r6/`；复用 R5 cache seed **37605**；**仅** inference schedule 变更

## 一句话

\[
\boxed{\texttt{joint\_inference\_insufficient}}
\]

有限步解耦 **未** 过 formal 门，但机制信息很强：**A1 第一步 \(n^{(1)}\) 将 P90 从 129° 压到 33°**（灾难性尾部大幅收缩）；**第二步 \(n^{(2)}\) 反而恶化**（median 20.0°）。A0 sanity 复现 R5（Δmedian **0.21°**）。

## Mechanism table（同 seed 37605）

| Branch / stage | median axis | P90 | median position |
|----------------|------------:|----:|----------------:|
| **A0** R5 joint | **16.85°** | **129.3°** | 2.27 cm |
| **A1** \(n^{(1)}\) | **17.13°** | **32.7°** | 2.20 cm |
| **A1** \(n^{(2)}\) formal | **19.98°** | **33.1°** | 2.45 cm |
| **A2** consensus | **18.90°** | **31.3°** | 2.33 cm |

## 机制解读

### A0 sanity ✓

\[
|\mathrm{median}_{A0}-\mathrm{median}_{R5}|=0.21^\circ\le0.5^\circ
\]

Joint path 未改。

### A1：decoupling 有效，但第二步 degradation

\[
n^{(1)}:\ 17.1^\circ\ /\ \mathbf{32.7^\circ\ P90}
\quad\Rightarrow\quad
\boxed{\text{固定 }p^{(0)}\text{ 切断 tail 有效}}
\]

\[
n^{(2)}:\ 20.0^\circ
\quad\Rightarrow\quad
\boxed{\text{第二次 center 更新把部分样本拉出正确 basin}}
\]

符合预注册关注的 **oscillation / second-step degradation** 情形。

### A2：consensus 收缩 tail，median 未过门

P90 **31.3°**（近 30° 门）；median **18.9°**。Top-8 center cluster 未形成足够 consensus 过 median 15° 门。

### \(D_{ho}\) diagnostic

- AUROC(\(-D_{ho}\), axis correct) = **0.37**（无判别力）
- \(\rho(S^{joint}, e)\approx -0.025\)；\(\rho(D_{ho}, e)\approx 0.005\)

**本格未** 将 \(D_{ho}\) 写入 score（`consistency_weight=0`）。

## Pattern

```text
pattern = joint_inference_insufficient
tags = [a1_failed, a2_failed, persistent_surface_next]
unlocks_o0e1_prereg = false
```

## Failure chain（完整）

```text
data → segmentation → cloud → B2 → visibility reference → finite joint inference
```

下一格：**R7 persistent surface / multi-frame reference**（证据链已足）。

## 措辞（硬）

| 能说 | 不能说 |
|------|--------|
| decoupling 显著压缩 P90 tail（129°→33°） | joint_inference_supported |
| \(n^{(2)}\) second-step degradation 实证 | alternating 已闭合 O0E0 |
| \(D_{ho}\) 不足以作独立判别（AUROC≈0.37） | 应在本格加 \(\lambda D_{ho}\) |
| R7 persistent surface 有充分依据 | O0E1 已解锁 |
