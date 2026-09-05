# RTWX-O0Q0R0C 预注册 — Native Event-Trace & Excitation Qualification

日期：2026-08-30  
状态：**正式冻结** `native_demo_action_replay_failure`  
依赖：O0Q0R0B=`precontact_branch_failure`  
报告：[`../../../REP/RTWX/world_perception/RTWX0O0Q0R0C_REPORT.md`](../../../REP/RTWX/world_perception/RTWX0O0Q0R0C_REPORT.md)  
**禁止**：修 mplib / 再写 scripted IK；改 \(H_{P0}=6\)；扫 \(I_x/I_z\)；restore P2/P3 接触态；23 格 nominal yaw；改 \(\tau=0.05\)；改 O0 target；开 O0Q0R1 / O0Q1 / O1；破坏 P1 regression。

## 唯一问题

> 官方同 seed 的 robot **qpos target** 能否作为冻结 \(A^*_{\mathrm{demo}}\)，自然进入 P2/P3，并在预接触根上确定性分叉？短窗 P0 的强惯量对照是否够敏感？

demo **只提供动作**，不提供 counterfactual state。事件全部由当前 simulator 在线检测。

## \(A^*_{\mathrm{demo}}\)

Seeds **0 / 1 / 2**，与 `seed.txt` + `episode_0000000–2` 对齐。  
hdf5 的 `action` qpos（若无则用观测 q 的移位）当作 `take_action` 的 qpos target。  
**不用** hdf5 object pose；**不用** IK。

从对应 reset 前向重放整条 trace，检测：

\[
t_{P2}=\min\{t:d_{\mathrm{EE,cup}}\le 0.08,\ \text{未稳定抓取}\},
\]

\[
t_{P3}=\min\{t:\text{爪}\le 0.25\text{ 且 }d<0.05\text{ 连续 }N=3\}.
\]

根：

\[
t_0=\max\{t<t_{P2}:d_t>0.12\}.
\]

正式分支只做

\[
x_{t_0}\xrightarrow{A^*_{t_0:}}.
\]

禁止 restore 接触态。

## P0 / P1（冻结，不回扫）

P0：\(H=6\) 个 \(K=5\) scene-step 组；无落地。  
Positive control **只作仪器**：\(I_x/I_z=\mathbf{4}\)；\(\omega=(2.0,0.3,1.5)\)（非主轴）。不是估计真实惯量。

P1：R0B 合同不动。若 identity 或摩擦 probe 坏掉 → `identity_regression_failure` / `contact_excitation_failure`。

## P2/P3 excitation

同一 \(A^*_{t_0:}\) 上，object-frame 各向异性摩擦（\(F=-8(Re_x\cdot v)Re_x\)，不加桌面初速）。只测 \(R_y90\)。

## Gates

| Gate | 条件 |
|------|------|
| G0-regression | P0 与 P1：\(D_H(I)<0.02\)；P0 窗口无接触 |
| G1-native-event | 每 seed 有 P2；\(n_{P3}\ge 2\) |
| G2-branch-repro | 预接触根 \(D_H(I)<0.02\)；两次 identity 的 \(\lvert\Delta t_{P2}\rvert,\lvert\Delta t_{P3}\rvert\le\delta_t=\mathbf{1}\) |
| G3-P0-excite | \(I_x/I_z=4\)：\(P(D_H(R_y90)>\tau)\ge 0.80\) |
| G4-P1-excite | 摩擦对照保持 \(P\ge 0.80\) |
| G5-P2/P3-excite | 接触各向异性：有样本的档 \(P\ge 0.80\) |

## Patterns（无 nominal yaw）

| Pattern | 条件 |
|--------|------|
| `identity_regression_failure` | ¬G0 |
| `native_demo_action_replay_failure` | G0 ∧ ¬G1 |
| `precontact_event_repro_failure` | G0 ∧ G1 ∧ ¬G2 |
| `freeflight_excitation_failure` | G0–G2 ∧ ¬G3 |
| `contact_excitation_failure` | G0–G3 ∧ ¬(G4 ∧ G5) |
| `counterfactual_instrument_qualified` | G0–G5 |

仅最后一项解锁 **O0Q0R1**。
