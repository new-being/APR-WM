# SYM-X 表示合同 — Adaptive Object State

日期：2026-08-30  
状态：**FROZEN（表示合同，非实验格）**  
依据：X0 `causal_symmetry_supported`；X1 `quotient_utility_supported`；X2 `gauge_reactivation_supported`  
**禁止**：开 SYM-X3 / RGB；改 X0–X2 门；把 \(E_{\mathrm{pred}}\) 写成结构充分性判据；声称参数/FLOPs 下降；把 task 相关写成撤销 world \(G\)。

## 已证明的机制链

\[
\boxed{
\text{发现因果对称}
\rightarrow
\text{在 }s/\hat G\text{ 上建模}
\rightarrow
\text{对称破缺时重新激活 }\gamma
}
\]

Host 仍是受控刚体 + oracle \(s\)。这是 **adaptive state-space topology**，不是静态商表示。

## 对象 memo

对物体 \(i\)：

\[
s_i=(\bar s_i,\gamma_i),\qquad \bar s_i=s_i/G_i,
\]

\[
m_i=\bigl(\bar s_i,\;\gamma_i,\;b_i(G)\bigr).
\]

\(b_i(G)=P(g\in G\mid D_{1:t})\)（X2 报 \(b=\exp(-D_H/\tau)\)）。档位

\[
\ell_i(\gamma)\in\{\mathrm{dormant},\mathrm{coarse},\mathrm{active}\}
\]

由 \(b\) 决定，而不是由单步预测残差决定。

World / task **分权**（X2 B3）：

\[
\alpha_{\gamma}^{\mathrm{world}},\qquad \alpha_{\gamma}^{\mathrm{task}}.
\]

B3 的读数是 \(\alpha^{\mathrm{world}}=0\)、\(\alpha^{\mathrm{task}}=1\)：动力学仍把 yaw 当 gauge，任务要 logo 朝向。  
**不得**因为 task 突然需要 \(\gamma\) 就整段撤销 world \(G\)。

与 TASK-XL 对齐：\(F_{\mathrm{physics}}\) 只吃 \(\alpha^{\mathrm{world}}\) 打开的坐标；\(q(A\mid\cdot)\) 可以吃 \(\alpha^{\mathrm{task}}\)。  
\(s^{\mathrm{phy}}\) **不是**固定 full \(R\)。

## 四层更新（正式）

\[
\boxed{
D_H
\rightarrow
b(G)
\rightarrow
\ell(\gamma)
\rightarrow
\text{representation allocation}
}
\]

**不是** \(E_{\mathrm{pred}}\uparrow\) 再决定加不加状态。

结构约束（X2 主读数，不是 G3 的字面）：

\[
\boxed{
\textbf{prediction error 不能作为结构充分性的唯一判据。}
}
\]

C4：\(E_{\mathrm{stale}}/E^{B0}=1.002\)，同时 \(D_H(R_y90)=6.44\gg\tau\)。  
可以同时出现 low predictive error **和** wrong causal abstraction。只有 counterfactual \(D_H(g)\) 能发现被商掉的自由度重新有因果作用。

## 三格各自钉住的命题

| 格 | 钉住 | 不钉住 |
|----|------|--------|
| X0 | \(\hat G\approx G_{\mathrm{causal}}\)；外形对称≠动力学对称；logo≠因果 | RGB 发现 \(G\) |
| X1 | \(\bar s=s/\hat G\) 预测非劣；表示**内容**减少 | MLP 宽度 / FLOPs 下降 |
| X2 | \(G_t\neq G_{t+1}\) 时可 \( \gamma:\mathrm{dormant}\to\mathrm{active}\)；task 与 world 分权 | 用 \(E\) 触发恢复 |

## 当前最强准确 claim

\[
\boxed{
\begin{aligned}
&\text{在受控刚体系统中，可从干预后的动力学等价性恢复 causal symmetry；}\\
&\text{由该 symmetry 构造的 quotient state 保持预测充分性；}\\
&\text{当 symmetry 被物理机制破坏时，可主动检测并重新激活被压缩的 gauge；}\\
&\text{task-only relevance 可以与 world-dynamics symmetry 分离。}
\end{aligned}
}
\]

## 明确未证

- quotient 显著降低参数或 FLOPs（X1 \(R_P(C0)=-1.09\)，非门、非正结果）
- RGB / perception 能自行发现 \(G\)
- 复杂真实物体上 \(G\) 一定是低维 Lie 群或简单有限群

## 对感知端的含义（X3 仍 LOCKED）

视觉下一步若开，目标**不是**重建固定 full state。  
应观测的是：维持 \(D_H\)、更新 \(b(G)\)、以及当前 \(\alpha^{\mathrm{world}/\mathrm{task}}\) 打开的坐标。  
O0G6A 已说明 shape matching 钉不住杯轴 yaw；SYM-X 说 yaw 是否进入 \(s^{\mathrm{phy}}\) 由因果对称决定，不是由外形决定。

**SYM-X3 继续 LOCKED。** 感知主线：\(\bar s=(p,n)\)，\(\gamma=\theta_{\mathrm{yaw}}\) 默认 dormant。O0A0：单帧不编码 yaw。O0E0P0 = `controlled_pose_observation_qualified`：可在 P0 cache 上重开 O0E0 science。O0E0 natural = 分布不够格，**B2 UNTESTED**。O0T0 **NOT RUN / DEPRIORITIZED**。O0Q* SIDE/FROZEN。


