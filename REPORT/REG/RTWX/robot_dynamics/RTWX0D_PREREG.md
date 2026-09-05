# RTWX-X0D 预注册 — Native Closed-Loop Structure Test

日期：2026-08-27  
状态：**已冻结**；**正式 RAN 2026-08-27**；**`low_order_structure_insufficient`**。报告：`REPORT/REP/RTWX/RTWX0D_REPORT.md`。  
依赖：X0C = `servo_signal_insufficient`（否掉“由 \((e_q,e_v)\) 重建有效关节力再接刚体”）。  
本单元：**不恢复隐藏力 \(\tau\)**；直接检验 native 闭环状态转移结构。  
不做：神经网络 / residual；TASK-XL；RGB；场景 \(\phi\)-ID；柜体接触；R10；容量 \(R_P\)；`set_qf`/`get_qf`/`qacc` buffer；不以 \(\ddot q\) 为主指标。

## 科学问题

\[
\boxed{\textbf{RTWX-X0D — Native Closed-Loop Structure Test}}
\]

\[
\boxed{
\text{不恢复隐藏力，只用 native }(q_t,\dot q_t,q_t^{tar})\rightarrow(q_{t+1},\dot q_{t+1}),
\text{受控机械系统是否仍存在比恒等更紧凑、可迁移的低阶结构坐标？}
}
\]

本单元是 **oracle 仿真器状态 / native-qpos 闭环结构诊断**，**不是**官方 RoboTwin 观测基准，**不是**物理参数辨识。\(\vartheta\) 称为 **effective closed-loop system coordinates**，不得称为 mass / inertia / torque gain。

X0D **不是**容量实验。仅 PASS 才解锁 **X0D1: structured vs latent capacity**。

## 合同

\[
x_t=\begin{bmatrix}q_t\\ \dot q_t\end{bmatrix},\qquad u_t=q_t^{tar},\qquad
x_{t+1}=F(x_t,u_t;\vartheta).
\]

主指标全部在 \((q,\dot q)\) 上。**不以 \(\ddot q\) 为主指标。**

规模与 X0C 相同（robotwin 正式冻结）：train/val/test = **48/24/24**，每段 **120** 次 native `take_action(..., action_type="qpos")`。自由空间、臂关节、夹爪排除、激励与 X0C 相同。numpy 仅单测可减小 \(n\)。

采集：复用 X0C 的 native 采集器（可记录 \(q_{\mathrm{post}},\dot q_{\mathrm{post}}\)）；禁止再开力矩通道。

## 模型（线性参数，禁止 MLP）

**M0 — Identity**

\[
\hat q_{t+1}=q_t,\qquad \hat{\dot q}_{t+1}=\dot q_t.
\]

**M1 — 独立 joint 二阶离散**

\[
\hat q_{t+1}=q_t+\Delta t\,\dot q_t,\qquad
\hat{\dot q}_{t+1,j}=a_j\dot q_{t,j}+b_j(q_{t,j}^{tar}-q_{t,j})+c_j.
\]

train 上逐关节 LS（含截距）。

**M2 — 跨关节线性闭环**

\[
\hat q_{t+1}=q_t+\Delta t\,\dot q_t,\qquad
\hat{\dot q}_{t+1}=A\dot q_t+B(q_t^{tar}-q_t)+c,\quad A,B\in\mathbb R^{d\times d}.
\]

**M3 — configuration-aware**

\[
\hat{\dot q}_{t+1}=A\dot q_t+B e_q+C\sin q_t+D\cos q_t+c.
\]

仍线性参数；无网络。

\(\Delta t=\Delta t_{\mathrm{control}}\)（逐步，与 X0C 相同）。

## 指标（冻结）

状态 \(x=(q,\dot q)\) 拼接。NRMSE 与既有 `_nrmse` 一致。

- \(E_1=\mathrm{NRMSE}(\hat x_{t+1},x_{t+1})\)
- 另报 \(E_1^q,E_1^{\dot q}\)
- 滚动：未来 \(q^{tar}\) 开环、状态闭环，\(H\in\{10,50\}\) → \(E_{\mathrm{roll}10},E_{\mathrm{roll}50}\)

## 门限（评分前冻结；见曲线后不得改）

| 门 | 规则 |
|----|------|
| **G0** transition nontrivial | train/val/test 均满足 \(E_1^{\mathrm{M0}}\ge 0.02\)，且 \(\mathrm{mean}_j\mathrm{std}(q_{t+1}-q_t)\ge 10^{-4}\)。否则 `copy_state_or_degenerate` → `instrument_failure` |
| **G1** structured one-step | 至少一个 \(k\in\{1,2,3\}\) 在 **test** 上 \(E_1^{M_k}\le 0.75\,E_1^{\mathrm{M0}}\) |
| **G2** rollout | 在 **val** 上 \(E_1\) 最低且满足 val-G1 的 \(k^\star\)：test 上 \(E_{\mathrm{roll}10}^{k^\star}<E_{\mathrm{roll}10}^{\mathrm{M0}}\)，且 \(E_{\mathrm{roll}50}^{k^\star}\) 有限并且 \(<10\max(E_{\mathrm{roll}50}^{\mathrm{M0}},10^{-8})\)（不爆炸） |
| **G3** transfer | \(k^\star\) 参数仅 train 拟合；\(E_1^{\mathrm{test}}\le 1.10\,E_1^{\mathrm{val}}\) |

若无模型过 val-G1，则 G1/G2 失败。

## Patterns

- `closed_loop_structure_supported`：G0–G3 过，且 M1 或 M2 在 test 上满足 G1 相对优势（不仅 M3）。
- `configuration_structure_required`：G0–G3 过，但 test 上仅 M3 满足 G1（M1、M2 均不满足）。
- `low_order_structure_insufficient`：G0 过但 G1 或 G2 不过。STOP 本低阶闭环结构族；不解锁 X0D1。
- `instrument_failure`：G0 失败。

`rtwx_x0d_passed` 仅当前三 pattern 中前两个且 G0–G3 全过。无容量声称。
