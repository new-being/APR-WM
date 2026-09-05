# RTWX-X0EH1 预注册 — Receding-Horizon Structured-vs-Neural Capacity

日期：2026-08-28  
状态：**已冻结（fresh 后）** — pattern `receding_structure_capacity_shift`  
依赖：X0EH = `short_horizon_sufficient`（\(H_{\mathrm{plan}}=16\)，\(K_{\mathrm{execute}}=4\)）。  
禁止：改 X0E/X0E1/X0ES/X0ES1/X0ES2/X0EH ledger；B2 residual 进 primary；旧 split 当 confirmatory。

## 问题

\[
\boxed{
\text{在统一 }K=4\text{ 真实观测 reset 部署合同下，冻结结构 M2 能否替代 generic neural capacity？}
\]

\(H_{\mathrm{plan}}=16\)（概念），\(K_{\mathrm{execute}}=4\)。所有模型同一 receding contract。

## 数据

seed **12601**；48/24/24 × 120 train/val/test；同 X0E native-qpos 激励。与 8601/9601/10601/11601 不相交。

## 模型（primary B0 vs B1）

- **B0**：Pure MLP，\(H\in\{8,16,32,64,128,256\}\)，同 X0E1 架构/训练协议
- **B1**：冻结 X0E-M2 basis，\(P=1752\)，train LS；无 residual

## 指标（test，\(K=4\)）

\(E_{\mathrm{exec}}^{K=4}\)，\(E_{\mathrm{end}}^{K=4}\)，\(r_{\mathrm{cat}}^{K=4}\)（catastrophe 定义不变）。

Reference：**H=256**。competent 当且仅当 \(r_{\mathrm{cat}}=0\) 且 \(E_{\mathrm{exec}}\le0.75\,E_{\mathrm{identity}}\)。

Matched vs ref：\(E_{\mathrm{exec}}\le1.05\,E_{\mathrm{exec}}^{\mathrm{ref}}\)，\(E_{\mathrm{end}}\le1.10\,E_{\mathrm{end}}^{\mathrm{ref}}\)，\(r_{\mathrm{cat}}=0\)。

\(P_{\mathrm{NN}}^{\min}=\min\{P_H:H\text{ matched}\}\)。若 B1 matched：\(R_P^{K=4}=1-P_{\mathrm{struct}}/P_{\mathrm{NN}}^{\min}\)，\(C_P^{K=4}=P_{\mathrm{NN}}^{\min}/1752\)。

另报 \(C_{\mathrm{deploy}}=(H_{\mathrm{plan}}\times\text{per-step forward cost})/K\)。

## Patterns

`receding_structure_capacity_shift` | `receding_structure_not_matched` | `reference_failure` | `no_capacity_gap`

`no_capacity_gap`：B1 matched 且 \(P_{\mathrm{NN}}^{\min}\le2.0\times1752\)。

仅 `receding_structure_capacity_shift` 允许写 \(R_P^{K=4}>0\)。
