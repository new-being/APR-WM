# RTWX-O0Q1 预注册 — Fresh Quotient Pose Perception

日期：2026-08-30  
状态：**LOCKED**（仅当 O0Q0R1=`task_yaw_causal_quotient_supported`）  
**禁止**：在 O0Q0R1 未 PASS 时改 O0 target；用 GT-\(R\) 做 \(\delta_O\) 修正；开 O1。

## 科学问题

> 在已证明 \(R\sim R R_y(\theta)\) 的任务动力学下，fresh 视觉能否恢复 \((p,n_{\mathrm{cup}})\)？

\[
\hat p^{\mathrm{pose}}=\hat p^{\mathrm{surf}}-\delta_y\hat n,\qquad
\delta_O=(0,\delta_y,0).
\]

Axis 门沿用 O0C：median \(e_{\mathrm{axis}}\le 15^\circ\)，P90 \(\le 30^\circ\)。必须有 axis-excitation（禁止永远竖直的 constant-up 伪 PASS）。

本文件 **不跑**。
