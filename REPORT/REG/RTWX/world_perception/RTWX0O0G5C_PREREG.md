# RTWX-O0G5C 预注册 — Fresh Canonical Identity Generalization

日期：2026-08-30  
状态：**已冻结**  
依赖：O0G5B=`global_canonical_identity_supported`（instrument / train-only）  
**禁止**：改 K / architecture / loss / quantization；RANSAC 或 Kabsch 作 primary；改 O0G5B 门限；arch sweep；pose rescue；解锁 O1 / O0C2。

## 科学问题

> 离散 CAD-anchor identity 到 **fresh episodes** 后，究竟是真正的视觉/几何泛化，还是训练位置记忆？

O0G5B 只支持：离散 CAD region identity **在训练集上可表示、可优化**。  
B1 略优于 B2，且 B1 含 \((u,v)\)，不能声称 “global context 是关键”。

本格只改变：

\[
\boxed{\text{train memorization}\rightarrow\text{fresh generalization}}
\]

## 冻结合同（不改任务，不改表示）

| 项 | 值 |
|----|------|
| Train | 复用 O0G5A cache seeds **31601 / 31602 / 31603** |
| Fresh test | **33601 / 33602 / 33603**；每 seed **12×120** |
| 分辨率 | **128²** RGB-D；`head` + `observer` |
| Anchors | **K=512** FPS，seed **32601**（同 O0G5B） |
| Quantization / InfoNCE / B1·B2 结构 | **同 O0G5B** |
| CAD | `visual/base0.glb` × `model_data0.json` scale |
| Primary matching | med \(e_{\mathrm{match}}^{norm}\le0.10\) **且** Top-5 \(\ge0.75\) |
| Confirmatory | P90；Top-1 label acc；Recall@\(0.05D_O\) / @\(0.10D_O\) |
| Mask strata | \(A<256\)；\(256\le A\le1024\)；\(A>1024\)（128² px） |
| B1-uv-shuffle | 打乱 test \((u,v)\)，保持 \(d,rgb\) |
| \(H(Y\mid U,V)\) | 8×8 image-cell；train vs fresh |
| **不做** | RANSAC；Kabsch primary；重选 \(K\) |

## 支路（都必须正式评 fresh）

### B1

\[
(u,v,d,rgb)\rightarrow\text{CAD anchor identity}
\]

### B2

\[
(u,v,d,rgb)+g_{\mathrm{object}}\rightarrow\text{CAD anchor identity}
\]

## 机制读数（预注册）

| 情况 | 读数 |
|------|------|
| A：B1 崩、B2 过 | 局部/像素位置记忆不能泛化；object-level context 提供 canonical identity |
| B：B1≈B2，都过 | global context **并非必要**；局部 RGBD+图像位置已有 transferable signal |
| C：B1≈B2，都崩 | O0G5B 主要是 memorization；**不开 O0G5R** |

若 fresh 上仍 `median≪0.1` 且 `P90∼0.7`：canonical identity 仍有约 10% 严重 mode error（与 direct orientation 重尾可能同构）。

## Gates

| Gate | 条件 |
|------|------|
| G0-support | agg \(P(N_{\mathrm{usable}}\ge64)\ge0.90\)；每 seed \(\ge0.85\) |
| B1 / B2 primary | med \(\le0.10\) **且** Top-5 \(\ge0.75\)（分别判定） |
| P90 / Top-1 / Recall@r | **confirmatory**，不改 primary 门 |

\[
\operatorname{Recall}@r=P\big(\|\hat a^O-x^{O,GT}\|\le r D_O\big),\quad r\in\{0.05,0.10\}
\]

## Patterns（互斥）

| Pattern | 条件 |
|--------|------|
| `coverage_failure` | ¬G0 |
| `canonical_identity_generalization_failure` | G0 ∧ ¬B1 ∧ ¬B2 |
| `local_identity_supported` | G0 ∧ B1 ∧ ¬B2 |
| `both_supported` | G0 ∧ B1 ∧ B2 |
| `global_context_supported` | G0 ∧ ¬B1 ∧ B2 |

`global_context_supported` **仅**当 B1 FAIL 且 B2 PASS。  
B1 已独立过门则不得声称 global 必需。

## 解锁

B1 或 B2 matching PASS → 允许预注册 **O0G5R**（imperfect matches + 全局几何一致性 → \(R\)）。  
`canonical_identity_generalization_failure` → O0G5R **LOCKED**。  
O0C2 / O1 **LOCKED**。本格停在 \(RGBD\to\) CAD canonical identity，**不做 pose rescue**。
