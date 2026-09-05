# RTWX-O0BEL0 预注册 — Relative Belief / Re-anchor Observability Audit

日期：2026-08-31  
状态：**已冻结 / RAN**（报告：`REPORT/REP/RTWX/world_perception/RTWX0O0BEL0_REPORT.md`）  
依赖：SEQ0 frozen chain；HYB0 deployable \(q_{\mathrm{rel}}\)；REL0 ICP  
**禁止**：re-anchor 执行；absolute A1；state 修改；ICP 调参；router

## 主假设

\[
H_{\rm belief}:\text{ relative registration 的在线 support history 含 accumulated error / impending failure 的可观测信息。}
\]

Shadow belief only：\(U_t\not\rightarrow\hat s_t\)。

## 数据

| 项 | 值 |
|----|-----|
| cal seed | **37610** |
| formal seed | **37611** |
| \(N_{\rm seq}\) | **100** + **100** |
| \(T\) | **64** transitions（65 frames） |
| motion | **完全继承 SEQ0**（仅延长 horizon） |

## Checkpoints

\(H\in\{1,2,4,8,16,32,64\}\)

## Belief（primary）

\[
U_t=\sum_{i=1}^{t}-\log(q_i+\epsilon),\quad \epsilon=10^{-6}
\]

\(q_t=\) HYB0 symmetric support（deployable）。

Diagnostics：\(\bar U_t=U_t/t\)；\(R_t=\max_{i=t-3..t}-\log(q_i+\epsilon)\)。

## Labels（evaluator only）

\[
z_t=1\iff e_n>15^\circ\lor e_p>5\text{ cm}
\]

\[
y_t^{(4)}=1\iff\exists k\in\{t+1,\ldots,t+4\}:z_k=1
\]

## Gates

1. **L1 replication**：\(H\le16\) chain 全 PASS；\(|median_{H16}-8.0^\circ|\le2^\circ\) → else `belief_sequence_replication_failure`
2. **L2 failure support**：\(r_F=P(\exists t:z_t=1)<0.10\) → `belief_failure_support_insufficient`（STOP L3）
3. **L3 observability**（cal）：\(AUROC(U_t,y^{(4)})\ge0.75\) 且 \(AUROC(U)-AUROC(t)\ge0.05\)；同 samples 比较 time baseline \(B_{\mathrm{time}}(t)=t\)

## Trigger calibration（L3 PASS 后，cal only）

precision \(\ge0.90\) 下最大化 useful recall；并列取最大 \(\tau_U\)。  
Formal 冻结 \(\tau_U^*\)，报告 precision / useful recall / median lead。

## Oracle ceiling（diagnostic only）

- adaptive GT reset at first shadow trigger
- periodic GT reset \(K\in\{8,16,32\}\)

## Patterns

- `belief_sequence_replication_failure`
- `belief_failure_support_insufficient`
- `relative_belief_insufficient`
- `relative_belief_observable` → `unlocks_sparse_reanchor_prereg`

## 实现

`rtwx_o0bel0.py`；`geometry/relative_belief.py`；`runs/rtwx_o0bel0/`
