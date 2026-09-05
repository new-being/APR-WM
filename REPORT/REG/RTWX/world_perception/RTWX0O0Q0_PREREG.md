# RTWX-O0Q0 预注册 — Task-Object Causal Quotient Audit

日期：2026-08-30  
状态：**正式冻结** `counterfactual_instrument_failure`  
依赖：O0G6A=`global_geometry_ambiguous`；SYM-X 表示合同（\(D_H\) 判因果对称，不是 \(E_{\mathrm{pred}}\)）  
**禁止**：RGB 进入 \(D_H^{RT}\)；改 O0G6A / O0C / SYM-X 门；把 O0 target 改成 \((p,n)\) **除非本格 PASS**；开 O0Q1 / O1 / O0G6R / O0C2 / SYM-X3；B3 摩擦破缺（本格只用 B2 惯量对照）。

## 科学问题

> RoboTwin 真实 `021_cup` 的杯轴 yaw，究竟只是「看不出来」，还是在 **任务动力学** 上也等价？

\[
\text{O0G6A}=\text{CAD shape 是否分得出 yaw},\qquad
\boxed{\text{O0Q0}=\text{真实 simulator dynamics 是否在乎 yaw}}
\]

视觉退化 **不得** 直接写成硬 quotient。PASS 才允许 \(s^{O,Q}_{\mathrm{pose}}=(p,n_{\mathrm{cup}})\)。FAIL 则 yaw 是

\[
\gamma_{\mathrm{yaw}}:\text{当前不可观测但可能有因果作用的 hidden/gauge belief}.
\]

## 反事实

从 snapshot \(x_t\) 出发，只改

\[
R_t'=R_t R_y(\theta),
\]

保持 cup \(p\)、world \(v\)、world \(\omega\)、robot \(q,\dot q\)、环境、后续 **已记录** 的 action 序列。  
比较 **quotient observables**（禁止 full-\(R\) 距离）：

\[
Y_t^Q=(p_t,\,n_t,\,v_t,\,\dot n_t,\,q_t,\,\dot q_t,\,J_t^{\mathrm{contact}}),
\quad
n_t=R_t e_y,\quad
\dot n_t=\omega_t\times n_t.
\]

\(J^{\mathrm{contact}}\)：含 cup 的接触 impulse 范数和。

\[
D_H^{RT}(g)=\mathrm{mean}_t\, d(Y^Q,Y^{Q,g}).
\]

无量纲 \(d\)（运行前冻结）：

\[
\frac{\|\Delta p\|}{0.05}+\frac{\|\Delta n\|}{0.20}+\frac{\|\Delta v\|}{0.30}+\frac{\|\Delta\dot n\|}{2.0}+\frac{\|\Delta q\|}{0.30}+\frac{\|\Delta\dot q\|}{1.5}+\frac{|\Delta J|}{0.02}.
\]

\(\tau=0.05\)。\(g=I\) 的数值地板必须 \(\ll\tau\)。

## 对象 / 动作源

| 项 | 冻结 |
|----|------|
| 任务 | `place_empty_cup`；cup=`021_cup` model 0 |
| seeds | **45101 / 45102 / 45103** |
| 规划 | mplib-screw fallback（与 TASK-X0 同 stub；不要求 CuRobo） |
| 动作 | **记录后重放**；不另训 policy |
| \(H\) | **8** 个 `take_action` |
| yaw 格 | \(\theta\in\{15^\circ,30^\circ,\ldots,345^\circ\}\)（**23** 非平凡；不含 \(I\)） |
| B2 | 同 mesh；\(I_x/I_z=\mathbf{1.5}\)（\(I_y\) 不变） |
| 每 seed | 最多 **3** 条 task episode + **1** 条 P0 激励 episode |
| 每 regime | 每 seed **1** 个 snapshot（缺则该 regime 无覆盖） |

## Regimes（各自 closure）

| ID | snapshot | 目的 |
|----|---------|------|
| P0 | 杯抬高 \(\ge 8\,\mathrm{cm}\)、给非轴向 \(\omega\)、机器人 hold | 惯量 |
| P1 | 桌面接触、夹爪远离 | 接触/摩擦 |
| P2 | 夹爪接近 / 轻触（\(\mathrm{dist}\in[0.02,0.12]\)） | 几何接触 |
| P3 | grasp / lift / place（夹爪近且杯离桌或接触冲量大） | 任务交互 |

P0 **单独**做激励 episode（任务演示里杯子一直在桌上则 P0 不会自然出现）。P2/P3 优先来自 `play_once` 记录；规划失败则用 **同一机器人 EE 邻域放置 / 闭爪 hold**（不另训 policy）。

## Bodies

- **B1 nominal**：真实 cup，不改惯量。
- **B2 anisotropic**：只改主成分 \(I_x\leftarrow 1.5\,I_z\)。外观/mesh/相机不动。

## Gates

| Gate | 条件 |
|------|------|
| G0-integrity | 所有 identity 重放 \(D_H(I)<0.02\)；restore+replay 闭合 |
| G1-excite | 每 regime 跨 seed \(\ge 3\) snapshot；B2：\(D_H(R_y90)>\tau\)，且 \(P(D_H>\tau\mid \theta\notin\{0,180\})\ge 0.80\) |
| G2-nominal | B1：\(P_{\theta\neq 0}(D_H\le\tau)\ge 0.90\)（23 格 × 有覆盖的 snapshot） |
| G3-regime | **每个** P0–P3：B1 上 \(P(D_H\le\tau)\ge 0.80\) |

## Patterns

| Pattern | 条件 |
|--------|------|
| `counterfactual_instrument_failure` | ¬G0 |
| `causal_symmetry_not_excited` | G0 ∧ ¬G1 |
| `task_yaw_causally_relevant` | G0 ∧ G1 ∧ ¬(G2 ∧ G3) |
| `task_yaw_causal_quotient_supported` | G0–G3 |

## 解锁

`task_yaw_causal_quotient_supported` → 允许预注册 **O0Q1**。  
本格实际：`counterfactual_instrument_failure` → **O0Q1 LOCKED**；O0 target 不改。  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0_REPORT.md)。
