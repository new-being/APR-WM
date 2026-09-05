# RTWX-X0ES 报告 — Macro-Transition Rollout Stability Audit

日期：2026-08-28  
状态：**正式冻结** \(\texttt{pathology\_not\_reproduced}\)  
预注册：`REPORT/REG/RTWX/RTWX0ES_PREREG.md`  
产物：`runs/rtwx_x0es/{run.json,metrics.json,header.json}`  
**不改** X0E / X0E1 账本。无重训、无新数据、无容量声称。

## 一句话

在冻结定义下（\(H=1\ldots50\)，catastrophic \(=\max E_H>10^6\) 或 nonfinite），**train 独有、val/test 干净** 这一 X0E1 印象 **不能复现**。

| split | 窗数 | catastrophic | 比例 |
|-------|------|----------------|------|
| train | 3360 | **2** | 0.060% |
| val | 1680 | **2** | 0.12% |
| test | 1680 | **7** | 0.42% |

因此 G0 失败，pattern = **`pathology_not_reproduced`**。  
不是“M2 在 val/test 永不炸”；是 **X0E1 的 \(H=10\) 均值被极少数窗主导**，而本格把同级灾难（\(E_H\to\infty\)）在 **三个 split 上都找到了**，test 计数甚至更高。

按预注册：**STOP 把机制写成 train-only**。不得据此改 X0E1。

## 仍可报告的伴随事实（不是改判 G1–G3）

G1–G3 在 G0 失败后 **不启用**。下列只作描述，不升级为 `support_excursion` / `local_instability`。

- **Teacher-forced vs free：** catastrophic 窗的 TF 一步中位误差 \(\approx 1.85\)（stable \(\approx 1.25\)），而 free 的 \(E_1\) 已到 6–12 并随 \(H\) 变 \(\infty\)。更像 **递归放大**，不是一步拟合整体失效。
- **谱半径：** train 上 cat 起点/发散前 \(\mathrm{median}(\rho_J)=2.26\)，stable \(0.79\)，\(P(\rho>1)_{\mathrm{cat}}=1\)。解析 Jacobian 与 FD 差 \(<10^{-6}\)。
- **OOD：** \(d_{\mathrm{OOD}}=Q_{99}(\mathrm{LOO\ NN})=232\)（robust-\(z\) 很松）。train cat 的 \(P(T_{\mathrm{OOD}}<T_{\mathrm{div}})=0\)（两次都是先/同时 \(T_{\mathrm{div}}\)）。
- **扰动：** cat 窗 \(A_{10}\) 中位数 \(\sim 10^{35}\)。
- **风险变量（train cat vs stable SMD）：** \(r_1=\|e_q\|\) \(4.15\)，\(r_2=\|\dot q\|\) \(2.02\)；\(r_3,r_4\) 弱。因 G0 失败，**不能**改判 `split_support_pathology`。

## 对 X0E1「23×」的含义

\(P_{\mathrm{M2}}=1752\) vs \(P_{\mathrm{NN}}^{\min}=40856\) 的 test 优势 **仍然不能写成 \(R_P\)**。  
本格进一步说明：\(H=10\) 上 “只有 train 均值爆炸” 是 **聚合假象**；拉长到 \(H=50\) 后，**所有 split 都有稀有递归灾难**。稳健容量替代仍不成立。下一步若做稳定性结构，必须针对 **稀有高 \(\rho_J\) 窗**，而不是假设问题仅存在于 train。

## 账本（X0E / X0E1 不改）

```text
RTWX-X0E  = PASS / nonlinear_increment_required     不改
RTWX-X0E1 = completed / structure_not_in_robust_set  NOT capacity PASS；不改
RTWX-X0ES = RAN / pathology_not_reproduced
            （H=50 catastrophic 在 train/val/test 均出现）
X0E2 / contraction / re-split = 未解锁
TASK-XL / R10 = LOCKED
```
