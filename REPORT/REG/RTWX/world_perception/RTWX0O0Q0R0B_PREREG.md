# RTWX-O0Q0R0B 预注册 — Pre-contact Branch Counterfactual Closure

日期：2026-08-30  
状态：**正式冻结** `precontact_branch_failure`  
依赖：O0Q0R0=`counterfactual_restore_failure`  
**禁止**：改 \(\tau=0.05\) / \(D_H(I)<0.02\)；RGB；瞬移进接触；**restore 已接触态**；扫 \(H\) 看 \(D_H\)；23 格 nominal yaw；改 O0 target；开 O0Q0R1 / O0Q1 / O1；各分支重跑 feedback / planner。

## 唯一问题

> 能否从 **identity 已闭合的接触前根状态** 分叉，用同一条冻结 open-loop \(A^*\) 让接触在前向仿真中自然生成，从而使 counterfactual 仪器闭合？

\[
\boxed{\text{不再 restore 接触态；只从确定性闭合的预接触根分叉}}
\]

本格 **不是** yaw 科学格。P1 作为已合格参考档，合同不变。

## 反事实结构

根 \(x_{t_0}\)：**无夹爪–杯接触**（杯可在桌上；这就是 R0 的 P1 态）。  
生成一次确定性 scripted 轨迹并冻结

\[
A^*=(a_{t_0},\ldots,a_{t_0+H-1})\quad\text{（qpos targets；生成后不再规划）}.
\]

所有分支：

\[
x_{t_0}\xrightarrow{A^*}Y,\qquad
gx_{t_0}\xrightarrow{A^*}Y^g.
\]

Identity 与 counterfactual 都重放同一 \(A^*\)。禁止 \(a_t^I\neq a_t^g\)。

## P2 / P3 = forward event，不是 restore 的 snapshot

| Regime | 定义（在 identity 的 \(A^*\) 上） |
|--------|------|
| P2 | 首次 \(d(\mathrm{EE},\mathrm{cup})<0.08\)（接近 / contact-onset） |
| P3 | 首次：爪 \(\le 0.25\) 且 \(d<0.05\) 连续 **\(N=3\)** 步；取窗口第一帧 |

测量窗口围绕该 event。根必须 \(\mathrm{dist}>0.12\)。

## P0 horizon（不扫 \(H\)）

\(z\) 向上，\(g=9.81\)。杯心相对桌面静高 \(h\)、竖直速度 \(v_z\)：

\[
h+v_zt-\tfrac12 gt^2=0,\qquad
t_{\mathrm{hit}}=\bigl(v_z+\sqrt{v_z^2+2gh}\bigr)/g.
\]

P0 **不**用 `take_action` TOPP（其时长与自由落体无关）。冻结：

\[
\Delta t=1/250,\quad K=5,\quad \Delta t_a=K\Delta t=0.02,\quad \Delta t_{\mathrm{margin}}=0.03.
\]

\[
H_{P0}=\bigl\lfloor(t_{\mathrm{hit}}-\Delta t_{\mathrm{margin}})/\Delta t_a\bigr\rfloor.
\]

整个 P0 窗口必须 **严格无接触**（\(J=0\) 且未落地）。\(H_{P0}<1\) 或窗口内出现接触 → `freeflight_horizon_failure`。

抬升 \(h=0.12\,\mathrm{m}\)，\(v_z=0\)，非轴向 \(\omega\)；夹爪远离。这是自由飞行初值，不是接触瞬移。

## \(A^*\) 生成（只用一次 planner）

最小确定性 scripted（不用 `play_once` / CuRobo）：

oracle 杯心 \(\to\) 固定接近 waypoint \(\to\) 闭爪。  
生成时可用 mplib IK 得到 qpos；**之后** identity / CF 只重放记录的 qpos。

## 激励（只测 \(R_y90\)）

\(\tau=0.05\)。P1 沿用 R0：\(v=(0.10,0,0)\)，\(F=-8(Re_x\cdot v)Re_x\)。  
P0：\(I_x\leftarrow 1.5 I_z\)。P2/P3：根上 COM 沿 \(e_x\) 偏 \(1.5\,\mathrm{cm}\)，然后走 \(A^*\)。

## Gates

| Gate | 条件 |
|------|------|
| G0-root | 每个入池根（P0 飞行根；P1/接触分支的预接触根）\(D_H(I)<0.02\) |
| G1-event | \(n_{P0},n_{P1},n_{P2},n_{P3}\ge 3\)；P2/P3 由 \(A^*\) 前向进入 |
| G2-repro | 两次 identity 的 P2/P3 event 下标 \(\lvert t^{(1)}-t^{(2)}\rvert\le\delta_t=\mathbf{1}\) |
| G3-excite | 每档 probe \(P(D_H(R_y90)>\tau)\ge 0.80\) |

## Patterns

| Pattern | 条件 |
|--------|------|
| `freeflight_horizon_failure` | P0：\(H_{P0}<1\)，或窗口内接触，或 P0 \(D_H(I)\ge 0.02\) |
| `precontact_branch_failure` | P0 合同成立，且 ¬(G0 其余根 ∧ G1 ∧ G2) |
| `regime_excitation_failure` | 前两门过且 ¬G3 |
| `counterfactual_instrument_qualified` | 全部 |

仅最后一项解锁 **O0Q0R1**。本格不写 yaw 科学 pattern。  
本格实际：`precontact_branch_failure` → **O0Q0R1 LOCKED**。  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0R0B_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0R0B_REPORT.md)。

## 刚体诊断（非门）

继续记录 \(I_B,c_{\mathrm{COM}},m\)。R0 已见 \(I_x\approx I_z\)；仪器闭合前 **不得**把旧 P0 nominal \(D_H\sim 1\) 读成惯量不对称。

Seeds：**0 / 1 / 2**。\(Y^Q\) 沿用 O0Q0。
