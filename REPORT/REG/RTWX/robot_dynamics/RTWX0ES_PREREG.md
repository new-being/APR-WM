# RTWX-X0ES 预注册 — Macro-Transition Rollout Stability Audit

日期：2026-08-28  
状态：**已冻结**；**正式 RAN 2026-08-28**；**`pathology_not_reproduced`**。报告：`REPORT/REP/RTWX/RTWX0ES_REPORT.md`。  
先前粗扫 `distribution_local_instability` **不是**本格结论。  
依赖：冻结 X0E-M2 与 X0E1 `cache_splits.npz`。  
X0E / X0E1 **账本不改**。本格 **不是** capacity；无 \(R_P\)。  
先前粗诊断 `distribution_local_instability` **不**作为本格结论；本格使用下列冻结 pattern。

## 禁止

不重拟合 basis；不改 X0E/X0E1 门；不训练 residual/MLP；不扩 width；不删坏轨迹后重报总体；不改 split；不重新采集；无 TASK-XL/RGB/R10。

只审计

\[
F_{M2}:(q_t,\dot q_t,q_t^{tar})\rightarrow(\Delta q_t,\Delta\dot q_t).
\]

## 问题

\[
\boxed{
\text{为什么冻结 M2 在 val/test 稳定且优于 reference，却在 train 上 }E_{\mathrm{roll}10}\sim 10^{37}\text{？}
}
\]

数据：`runs/rtwx_x0e1/cache_splits.npz`。\(W\) 仅 train 上按冻结 \(\phi\) LS。

## 统计（每 split、每 episode、每起点 \(t\)，\(H=1,\ldots,50\)）

- \(E_H=\|\hat x_{t+H}-x_{t+H}\|_2/\mathrm{scale}(x)\)，\(\mathrm{scale}=\mathrm{rms}(x_{\mathrm{train}})\)。
- \(T_{\mathrm{div}}=\min\{H:E_H>10\}\)，否则 \(+\infty\)。
- catastrophic：\(\max_H E_H>10^6\) 或 nonfinite。
- \(z=[q,\dot q,e_q]\) robust 标准化（median/IQR，IQR 过小则置 1）。
- \(d_{\mathrm{OOD}}=\) train leave-one-out NN 距离的 99 分位。
- \(T_{\mathrm{OOD}}=\min\{k:d_{\mathrm{NN}}(\hat z_{t+k})>d_{\mathrm{OOD}}\}\)（\(k=0\) 为起点）。
- \(\Delta T=T_{\mathrm{div}}-T_{\mathrm{OOD}}\)。
- \(J=\partial\hat x_{t+1}/\partial x_t\)（解析；抽检 FD），\(\rho_J=\rho(J)\)。
- 扰动 \(A_H\)，\(H\in\{1,5,10\}\)，\(\varepsilon=10^{-3}\times\) 各维 scale，\(\pm\) 坐标轴。
- Teacher-forced vs free。
- 预注册风险：\(r_1=\|e_q\|_2,\;r_2=\|\dot q\|_2,\;r_3=\|\Delta q^{tar}\|_2,\;r_4=\) 到经验关节包络的逆距离。

## Gates / patterns

- **G0** train 存在 catastrophic，val/test 无同级 → 否则 `pathology_not_reproduced`
- **G1** catastrophic 中 \(P(T_{\mathrm{OOD}}<T_{\mathrm{div}})\ge 0.8\) → `support_excursion`
- **G2** G1 否：cat 相对 stable \(\mathrm{median}(\rho_J)\) 差 \(\ge 0.10\) 且 \(P(\rho_J>1)_{\mathrm{cat}}\ge 0.5\) → `local_instability`
- **G3** G1/G2 否：某一 \(r_i\) 的 |标准化中位数差| \(\ge 1\)，且该尾部主要在 train → `split_support_pathology`
- 否则 `instability_unresolved`

只允许上述五类。不解锁 capacity。仅诊断结果可解锁后续 **稳定性专用** 新格（X0E2 / 收缩结构 / difficulty-balanced split）。
