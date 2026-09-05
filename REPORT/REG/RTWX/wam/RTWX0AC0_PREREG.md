# RTWX-AC0 预注册 — Explicit-State Action-Chunk Provisioning

日期：2026-09-01  
状态：**已冻结**；正式跑 **INTERIM**（见 `REPORT/REP/RTWX/wam/RTWX0AC0_REPORT.md`）  
依赖：MR0-P0 contract \(H_a=8,d_a=14,\texttt{joint\_position}\)；三任务 demo_clean hdf5  
**禁止**：RGB；diffusion；CVAE；Transformer bakeoff；\(t/T\)；window 级随机划分；改 action 表示为 delta/EE；看 bakeoff 再改 G3 阈值；TASK-X1

## 问题

\[
(s_t,z_g)\ \text{vs}\ (s_t,z_g,z_t^p)
\quad\text{能否用同一小型确定性 MLP 监督出 }A_{t:t+7}\text{？}
\]

`state_source = oracle_structured`  
`scientific_scope = action_representation_only`  
三任务同一合同，禁止混用 GT / learned。

## Branches（\(\Delta=1\)：仅 process 特征）

- **B0** \(f(s,z_g)\)
- **B1** \(f(s,z_g,z^p)\)，\(z^p=[d_{EE,O},c_{grasp},d_{O,G},c_{contact},c_{goal}]\) 由当前状态解析，无训练 process 模块

## 数据

每任务 50 demo；episode 先 40/5/5 再切窗；不足 8 步的尾窗丢弃。标签 = hdf5 `action` joint_position \(8\times14\)。

## 训练

256×3 GELU MLP；AdamW \(3\times10^{-4}\)；L1 on chunk；val L1 选 ckpt；max 100 ep / patience 10。

## Stage B bakeoff

\(N=16\)/task；seeds 从 **50601** 起（避开 P0 G3 的 38601）；\(K_a=1\) 只执行 \(A_t[0]\)。

Winner lex：pooled success → min-task success → 低 safety → 低 val L1。  
若 B1 pooled 优势 \(<5\) pp 且 min-task 不高过 B0 → **B0**（complexity tie）。

## 冻结后

winner → `runs/rtwx_mr0_p0/frozen_chunk_policy.pt`，**原样重跑 P0**（G3 不放宽）。
