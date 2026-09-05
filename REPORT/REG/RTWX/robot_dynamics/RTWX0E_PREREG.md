# RTWX-X0E 预注册 — Direct Increment Representation

日期：2026-08-28  
状态：**已冻结**；**正式 RAN 2026-08-28**；**`nonlinear_increment_required`**。报告：`REPORT/REP/RTWX/RTWX0E_REPORT.md`。  
依赖：X0C = `servo_signal_insufficient`；X0D = **`low_order_structure_insufficient`（已冻结，禁止改 M1–M3）**。  
本单元是 **新族**，不是 X0D 补丁，不是 physics closure。  
不做：大 MLP；diffusion；neural residual（本格）；TASK-XL；RGB；\(\phi\)-ID；柜体接触；R10；容量 \(R_P\)；`set_qf`/`get_qf`；Euler \(\Delta q=\Delta t\dot q\)。

## 科学问题

\[
\boxed{\textbf{RTWX-X0E — Direct Increment Representation}}
\]

\[
\boxed{
\text{native action window 本身是否存在低维、可学习、稳定的 transition representation？}
}
\]

基本因果时间尺度是 **一次 native `take_action`（qpos）窗口**，不是 PhysX substep，也不是外部 Euler 时钟。

直接预测

\[
\Delta q_t=q_{t+1}-q_t,\qquad
\Delta\dot q_t=\dot q_{t+1}-\dot q_t,
\]

输入

\[
(q_t,\dot q_t,q_t^{tar}).
\]

**禁止** \(\Delta q=\Delta t_{\mathrm{control}}\dot q\)。

\(\vartheta\) = **native-window increment coordinates**，不得称为 mass / inertia / torque gain。

本单元是 oracle 仿真器状态 / native-qpos 诊断，**不是**官方观测基准。X0E **不是**容量实验。仅 PASS 才解锁 **X0E1: increment vs latent/residual capacity**。

## 规模（robotwin 正式冻结）

与 X0C/X0D 相同：train/val/test = **48/24/24**，每段 **120** 次 native `take_action(..., action_type="qpos")`。自由空间、臂关节、夹爪排除、激励与采集器同 X0C。numpy 仅单测可减小 \(n\)。

## 模型（禁止 MLP / diffusion）

\(x=(q,\dot q)\)，\(e_q=q^{tar}-q\)，\(s=(q,\dot q)\)。

**M0 — Identity**

\[
\widehat{\Delta x}=0 \quad\Rightarrow\quad \hat q_{t+1}=q_t,\;\hat{\dot q}_{t+1}=\dot q_t.
\]

**M1 — linear increment**

\[
\widehat{\Delta x}
=
W\begin{bmatrix}s\\ e_q\end{bmatrix}+b.
\]

**M2 — small nonlinear structured basis + linear head**

固定基

\[
\phi=
[q,\;\dot q,\;e_q,\;\sin q,\;\cos q,\;e_q\odot|e_q|],
\]

\[
\widehat{\Delta x}=W\phi+b.
\]

无网络。参数仅 train 最小二乘。

## 指标（冻结）

- \(E_1=\mathrm{NRMSE}(\hat x_{t+1},x_{t+1})\)，另报 \(E_1^q,E_1^{\dot q}\)
- 滚动：未来 \(q^{tar}\) 开环、状态闭环（\(x\leftarrow x+\widehat{\Delta x}\)），\(H\in\{10,50\}\)

NRMSE 与既有 `_nrmse` 一致。

## 门限（评分前冻结；见曲线后不得改）

| 门 | 规则 |
|----|------|
| **G0** nontrivial | train/val/test 均 \(E_1^{M0}\ge 0.02\)，且 \(\mathrm{mean}_j\mathrm{std}(\Delta q)\ge 10^{-4}\)。否则 `instrument_failure` |
| **G1** one-step | 至少一个 \(k\in\{1,2\}\) 在 **test** 上 \(E_1^{M_k}\le 0.75\,E_1^{M0}\) |
| **G2** rollout | val 上 \(E_1\) 最低且过 val-G1 的 \(k^\star\)：test \(E_{\mathrm{roll}10}^{k^\star}<E_{\mathrm{roll}10}^{M0}\)，且 \(E_{\mathrm{roll}50}^{k^\star}\) 有限且 \(<10\max(E_{\mathrm{roll}50}^{M0},10^{-8})\) |
| **G3** transfer | \(k^\star\) 仅 train 拟合；\(E_1^{\mathrm{test}}\le 1.10\,E_1^{\mathrm{val}}\) |

若 val 无模型过 G1，则 G1/G2 失败。

## Patterns

- `native_window_increment_supported`：G0–G3 过，且 M1 在 test 上满足 G1（M2 可同时过）。
- `nonlinear_increment_required`：G0–G3 过，但 test 上仅 M2 满足 G1。
- `low_capacity_increment_insufficient`：G0 过但 G1 或 G2 不过。**STOP** 本低容量显式 increment 族；此后才允许开 neural residual / latent baseline。
- `instrument_failure`：G0 失败。

`rtwx_x0e_passed` 仅当前两个 PASS pattern。无容量声称。
