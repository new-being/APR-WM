# RTWX-X0E1 预注册 — Structured vs latent capacity（native-window increment）

日期：2026-08-28  
状态：**已冻结**；**正式 RAN 2026-08-28**；**`structure_not_in_robust_set`**。报告：`REPORT/REP/RTWX/RTWX0E1_REPORT.md`。  
依赖：X0E = **PASS / `nonlinear_increment_required`**（**禁止回头改 X0E 门或 basis**）。  
X0C hidden-force STOP；X0D Euler 闭环族 STOP。  
本格回答 CAP-X 主线在 RoboTwin native-qpos 上的容量问题，**不是**更复杂结构发明。  
不做：diffusion；改 X0E-M2 基；TASK-XL；RGB；R10；力通道。

## 科学问题

\[
\boxed{\textbf{RTWX-X0E1 — 该冻结结构能替代多少 generic learned capacity？}}
\]

合同：macro-transition \(\Delta x=(\Delta q,\Delta\dot q)\)，时间尺度 = native `take_action` 窗口。  
\(\vartheta\) 仍是 native-window increment coordinates，**不是** torque-level physics。

## 三族（冻结，禁止中途改架构）

**B0 — Pure latent / neural**

\[
F_\psi(q,\dot q,q^{tar})\rightarrow(\Delta q,\Delta\dot q)
\]

三层 SiLU MLP，宽 \(H\in\{8,16,32,64,128,256\}\)，线性头。输入 \((q,\dot q,q^{tar})\)（12+12+12）。

**B1 — Frozen X0E-M2**

\[
\phi=[q,\dot q,e_q,\sin q,\cos q,e_q\odot|e_q|],\quad
\widehat{\Delta x}=W\phi+b
\]

**禁止重选 basis。** \(W\) 仅在 train 上 LS，与 X0E 相同。

**B2 — M2 + residual**

\[
\Delta x=W\phi(x,u)+R_\psi(q,\dot q,q^{tar})
\]

\(W\phi\) 冻结为 B1；残差同族三层 SiLU，\(H_r\in\{4,8,16,32,64\}\)。

## 数据

与 X0E 同一生成过程：seed **8601**，48/24/24 × 120 native qpos，自由空间臂关节。numpy 仅单测可减小 \(n\) 与宽度网格。

## 训练（所有可学习宽度共用，禁止按宽度调参）

| 项 | 冻结 |
|----|------|
| 优化器 | AdamW，lr \(10^{-3}\)，wd \(10^{-4}\) |
| batch | 256 |
| epochs | 40，val \(E_1\) patience 8，restore best |
| 输入标准化 | train mean/std，所有模型共用 |
| 种子 | \(\{201,202,203,204,205\}\)；B1 无随机种子（LS 唯一） |

## 指标

\(E_1,E_{\mathrm{roll}10},E_{\mathrm{roll}50}\) 在 \((q,\dot q)\) 上，定义同 X0E。报告参数量 \(P\)。

## \(G_{\mathrm{robust}}\)（X0E1 新协议，不改 X0E PASS）

参与容量声称的每个模型，**train/val/test 全部**：

- \(E_1,E_{\mathrm{roll}10},E_{\mathrm{roll}50}\) 有限、无 NaN/Inf；
- \(E_{\mathrm{roll}10}<10\)，\(E_{\mathrm{roll}50}<20\)。

某一 split 出现 \(10^{37}\) 类灾难则 **不得** 进入“稳定容量替代”结论。

## Reference 与 competence

\[
F_{\mathrm{ref}}=\text{B0 }H=256
\]

（5 种子 test 均值）。

\[
\tau_1=0.90\,E_1^{\mathrm{M0,test}},\qquad
\tau_{10}=0.90\,E_{\mathrm{roll}10}^{\mathrm{M0,test}}.
\]

**G_comp**：\(E_{1,\mathrm{ref}}<\tau_1\) 且 \(E_{\mathrm{roll}10,\mathrm{ref}}<\tau_{10}\) 且 ref 过 \(G_{\mathrm{robust}}\)。  
否则 pattern = **`reference_failure`**，\(R_P\) **undefined**（CAP-X2 教训）。

## Matched（相对 ref 均值）

\[
E_1\le 1.05\,E_{1,\mathrm{ref}},\quad
E_{\mathrm{roll}10}\le 1.10\,E_{\mathrm{roll}10,\mathrm{ref}},\quad
E_{\mathrm{roll}50}\le 1.10\,E_{\mathrm{roll}50,\mathrm{ref}},
\]

且该模型过 \(G_{\mathrm{robust}}\)。

\[
P_{\mathrm{NN}}^{\min}=\min\{H\in B0:\text{matched}\},\qquad
P_{\mathrm{res}}^{\min}=\min\{H_r\in B2:\text{matched}\}.
\]

若 B1 matched：`structure_only_matched=true`。

\[
R_P^{\mathrm{struct}}=1-\frac{P(\mathrm{B1})}{P(B0,P_{\mathrm{NN}}^{\min})}
\quad\text{（仅当 B1 与 }P_{\mathrm{NN}}^{\min}\text{ 均定义）}
\]

\[
R_P^{\mathrm{res}}=1-\frac{P(B2,P_{\mathrm{res}}^{\min})}{P(B0,P_{\mathrm{NN}}^{\min})}
\quad\text{（仅当两者均定义）}
\]

无 matched 的族记 `undefined`，不得事后改门槛。

## Patterns

- `capacity_substitution_supported`：G_comp 过，且 \(P_{\mathrm{NN}}^{\min}\) 有定义，且（B1 matched 或 \(P_{\mathrm{res}}^{\min}\) 有定义）。
- `structure_not_in_robust_set`：G_comp 过，\(P_{\mathrm{NN}}^{\min}\) 有定义，但 B1 未过 \(G_{\mathrm{robust}}\) 或未 matched；B2 可另报。
- `neural_no_match`：G_comp 过但没有任何 B0 宽度 matched（含 H=256 应自匹配；若不自匹配则实现错误或 ref 聚合问题 → `instrument_failure`）。
- `reference_failure`：G_comp 失败。
- `instrument_failure`：数据/X0E 依赖失败。

`rtwx_x0e1_passed` 仅当 pattern 为 `capacity_substitution_supported` 或 `structure_not_in_robust_set`（后者允许对 **残差/纯网络** 做容量叙述，但 **不得** 声称冻结 M2 单独构成稳定容量替代）。

无 R10 解锁。
