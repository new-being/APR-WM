# RTWX-O0Q0R0 预注册 — Native Counterfactual Instrument Closure

日期：2026-08-30  
状态：**正式冻结** `counterfactual_restore_failure`  
依赖：O0Q0=`counterfactual_instrument_failure`；O0G6A=`global_geometry_ambiguous`；SYM-X 表示合同  
**禁止**：重跑 O0Q0；改 \(\tau=0.05\) / \(D_H(I)<0.02\)；RGB 进入 \(D_H\)；object / EE **瞬移进接触**；在本格对 nominal yaw 下科学结论；改 O0 target；开 O0Q0R1 / O0Q1 / O1 / O0G6R / O0C2 / SYM-X3；identity 后再挑“安静” snapshot。

## 唯一问题

> 能否获得「真实走到那里」的 P0–P3 snapshot，使 restore+replay 本身确定性闭合？

\[
\boxed{\text{O0Q0R0}=\text{native counterfactual instrument}}
\]

本格 **不是** causal-yaw 科学格。不对 \(D_H^{RT}(R_y)\) 在 nominal `021_cup` 上做 G2/G3。

## 禁止的 snapshot 构造

O0Q0 的 P2/P3 把杯子塞进 EE 邻域 / 闭爪内部，接触求解器进入穿透区，于是

\[
D_H(I)\gg 0.
\]

冻结：

| Regime | 合法来源 |
|--------|----------|
| P0 | 自由飞行 **初值**（抬高、非轴向 \(\omega\)、夹爪远离）；**不是**接触瞬移 |
| P1 | 合法 reset 的桌面态，或前向轨迹上首次满足桌上规则的帧 |
| P2 / P3 | 必须从合法 reset **前向运行进入** |

优先：官方 `place_empty_cup` hdf5 的原生 qpos，**必须与 `seed.txt` 对齐**（禁止把 episode_0 接到无关 seed 上）。  
正式 seed：**0 / 1 / 2**（`seed.txt` 前三，对应 `episode_0000000–2`）。  
其次：`play_once`（mplib fallback 允许）。  
否则：本文件冻结的确定性 scripted controller（`take_action` / `take_dense_action`）。  
**不允许** `set_pose(cup)` 到 EE 附近或爪内。

只有前向动力学生成的 contact state 才能进入 counterfactual pool。

## Snapshot 选择（运行前冻结；先选后测 identity）

每个 seed 的每条前向轨迹上，**第一次**满足者入池；缺 horizon 则该次不算。禁止按 \(D_H(I)\) 回挑。

| ID | 规则 |
|----|------|
| P0 | 专用自由飞行 episode 的首帧：\(z\ge z_0+8\,\mathrm{cm}\)、\(\|\omega\times e_y\|>0.1\)、\(\mathrm{dist}(\mathrm{EE},\mathrm{cup})>0.15\) |
| P1 | 首次：\(z\le z_0+1.5\,\mathrm{cm}\)、两臂 EE–cup \(>0.15\)、\(\|\omega\|<0.5\) |
| P2 | 首次：活动臂 EE–cup \(\in[0.02,0.12]\)、该爪打开（\(\ge 0.35\)）、尚未稳定抓取 |
| P3 | 首次：该爪关闭（\(\le 0.25\)）且 cup 接触持续 **\(N=3\)** 个已记录控制步；取该窗口第一帧 |

若某 regime 选不出 snapshot，**不得删掉该 regime 后继续宣称仪器合格**。

## Identity

每个入池 \(s_j\) 至少两次：

\[
s_j\xrightarrow{\mathrm{restore}} a_{j:j+H}
\quad\text{与}\quad
s_j\xrightarrow{\mathrm{restore}} a_{j:j+H}.
\]

控制通道必须与记录时相同（`take_action` 或 dense qpos 路径）。  
\(H=8\)。\(Y^Q\) 与 \(d\) 沿用 O0Q0。**不改阈值：**

\[
\boxed{D_H(I)<0.02}
\]

## 激励合同（G1；按 regime，不是同一 B2 惯量套所有档）

\(\tau=0.05\)。目标：证明 **这个 regime 的 probe 有能力发现 yaw-dependent physics**。只测 \(R_y(90^\circ)\)，不扫 23 格。

\[
\boxed{
\begin{array}{ll}
\mathrm{P0}:& I_x\leftarrow 1.5\,I_z\quad\text{（空中已有非轴向 }\omega\text{）}\\
\mathrm{P1}:& \text{各向异性摩擦：初速 }v=(0.10,0,0),\quad F=-8\,(Re_x\cdot v)\,Re_x\\
\mathrm{P2/P3}:& \text{COM 沿体轴 }e_x\text{ 偏 }1.5\,\mathrm{cm}
\end{array}}
\]

Probe 只在 G1 滚动中施加，G0 identity 用名义刚体参数。每次滚动后恢复 \(I\)、COM。

## Gates

| Gate | 条件 |
|------|------|
| G0-restore | 所有**已入池** snapshot：\(D_H(I)<0.02\) |
| G-native | P0–P3 各跨 seed \(\ge 3\) snapshot；P2/P3 来源 \(\in\{\mathrm{hdf5},\mathrm{play\_once},\mathrm{scripted}\}\) |
| G1-excite | 每个 regime：该档 probe 的 \(P(D_H(R_y90)>\tau)\ge 0.80\) |

## Patterns（仅仪器）

| Pattern | 条件 |
|--------|------|
| `counterfactual_restore_failure` | 已入池但 ¬G0 |
| `native_contact_snapshot_failure` | G0 对已入池成立（或无入池）且 ¬G-native |
| `regime_excitation_failure` | G0 ∧ G-native ∧ ¬G1 |
| `counterfactual_instrument_qualified` | G0 ∧ G-native ∧ G1 |

若某接触档根本走不到合法态，记入 `native_contact_snapshot_failure`（即 unavailable），**不要**删档后做 yaw claim。

## P0 刚体参数（诊断，非门）

记录 \(I_B,\,c_{\mathrm{COM}},\,m\)，检查 \(g I_B g^\top \stackrel{?}{=} I_B\)（尤其 \(I_x\stackrel{?}{=}I_z\)）。  
CAD 轴对称但惯量不对称 → 对应 SYM-X0 C4；**只解释，不构成 yaw 科学结论。**

## 解锁

仅 `counterfactual_instrument_qualified` → 允许预注册 **O0Q0R1**（fresh causal-yaw）。  
本格实际：`counterfactual_restore_failure` → **O0Q0R1 LOCKED**。  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0R0_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0R0_REPORT.md)。
