# SYM-X Preregistration — Causal Symmetry Discovery and Quotient State

日期：2026-08-30  
状态：**FROZEN（family）**；**SYM-X0 / SYM-X1 / SYM-X2 RAN**；**X3 LOCKED**  
依赖：RTWX-O0G6A=`global_geometry_ambiguous`（刚体 cup：shape matching 不能钉住 yaw）  
**不**混入 RGB 感知；**不**改 O0G6A/O0G5* 门；**不**开 O0G6R；R10 / O1 LOCKED。

## 为何独立成线

O0G6A 测的是 **shape matching**。本线测的是：

\[
\boxed{
\text{干预后的动力学等价}\ \Rightarrow\ \text{可否从 world-state 删除该自由度}
}
\]

\[
\boxed{
\text{O0G6A = shape};\qquad
\text{SYM-X0 = intervention / causal symmetry.}
}
\]

不把 RGB、latent WM、ICP 塞进第一格。

## 核心命题

\[
\boxed{
\text{模型能否通过动力学等价性自动发现 }G,
\text{并把状态压到等价类而不损失预测/控制能力？}
}
\]

## 格子（顺序冻结）

```text
SYM-X0  Causal Symmetry Discovery     oracle s + counterfactual interventions → 能否发现 G？
SYM-X1  Quotient Representation Utility  full vs oracle G* vs discovered Ĝ vs unconstrained latent
SYM-X2  Symmetry Breaking & Reactivation   破坏几何/惯量/任务 → 撤销错误 quotient、激活 gauge
SYM-X3  Perception-Mediated Discovery    RGBD → belief over G   【LOCKED】
```

X3 明确锁住：先证明方法本身，再替换 oracle 视觉。

## X1 / X2 只作指针（本文件不跑）

**X1**：**RAN / `quotient_utility_supported`**。B2 相对 B0 非劣；\(s/\hat G\) 不含 yaw；B3 未自发 quotient；该 MLP 网格 \(R_P\) 非正。

**X2**：**RAN / `gauge_reactivation_supported`**。物理破缺 \(T_{\mathrm{revoke}}=3\)；false revoke 0/20；B3 不撤 world \(G\)。检测靠 \(D_H\)，不是 MLP 单步 \(E\)。

## 与 RTWX 的边界

| 线 | 问题 |
|----|------|
| O0G6A | 可见形状能否唯一确定 full \(R\) |
| **SYM-X** | 动力学是否把某些 \(g\) 当成同一机制 |
| O0 / O1 | RGB→\(s^O\) / filter（仍 LOCKED） |

不得把 SYM-X0 写成 “O0G6B”。

## 表示合同（X0–X2 已闭链后）

实验格已闭。正式对象状态见 [`SYMX_MECHANISM.md`](SYMX_MECHANISM.md)。不在本文件实现，不开 X3。

\[
m_i=(\bar s_i,\gamma_i,b_i(G)),\qquad
D_H\to b(G)\to\ell(\gamma)\to\text{allocation}.
\]

\(F_{\mathrm{physics}}\) 跟 \(\alpha_{\gamma}^{\mathrm{world}}\)；task 跟 \(\alpha_{\gamma}^{\mathrm{task}}\)。  
**prediction error 不是结构充分性的唯一判据。**
