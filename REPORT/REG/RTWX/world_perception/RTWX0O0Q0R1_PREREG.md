# RTWX-O0Q0R1 预注册 — Native Cup Causal-Yaw Audit

日期：2026-08-30  
状态：**LOCKED**（仅当 O0Q0R0E=`native_dense_trace_qualified`）  
**禁止**：在 R0E 未合格时开本格；改 \(\tau=0.05\)；瞬移进接触；restore 接触态；改 O0 target（除非本格 PASS）；开 O0Q1 / O1。

## 科学问题

> 在 **已闭合** 的 native counterfactual 仪器上，nominal `021_cup` 的杯轴 yaw 是否被任务动力学拒绝？

本文件 **不跑**。R0E 合格后再冻结 runner。

## 预告（R0 合格后才生效）

- 23 个 frozen yaw（\(15^\circ\ldots345^\circ\)）；不含 \(I\)
- nominal `021_cup`；P0–P3 native snapshots（沿用 R0 选择规则）
- \(Y^Q\)、\(D_H^{RT}\)、\(\tau=0.05\)
- regime-specific excitation 只作仪器对照，不替代 nominal 门

| Pattern | 含义 |
|--------|------|
| `task_yaw_causally_relevant` | 有效 regime 上 nominal \(D_H(R_y)>\tau\) 稳定出现 → O0 保持 full \(T,R\)；O1 方向是 \(p(\gamma_{\mathrm{yaw}}\mid\mathrm{history},\mathrm{interaction})\) |
| `task_yaw_causal_quotient_supported` | nominal yaw 全部等价 → 才解锁 O0Q1，才有资格讨论 perception target 改成 quotient |

未合格仪器时这两个 pattern **都不写**。
