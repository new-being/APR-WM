# SYM-X0 预注册 — Causal Symmetry Discovery

日期：2026-08-30  
状态：**已冻结；本格 RAN / `causal_symmetry_supported`**  
依赖：[`SYMX_PREREG.md`](SYMX_PREREG.md)；O0G6A shape 结果仅作动机，**不**作本格门限  
**禁止**：RGB 进入 \(D_H\)；训练 latent；改 O0G*；解锁 X1 以外的格子；R10 / O1。

## 科学问题

> 在 oracle 刚体状态上，仅用 counterfactual 短 rollout，能否自动发现因果对称 \(G^*\)——并且 **C4 不误 quotient、C5 不被 appearance 欺骗**？

\[
D_H(g)
=
\mathbb E_{s,a}
\left[
d\!\left(
g^{-1}F^H(gs,T_ga),\;
F^H(s,a)
\right)
\right]
\]

\(D_H(g)\approx0\) ⇒ \(g\) 是同一物理机制的不同标签。

## 仪器（无 RGB）

Host：`symx_rigid.v1`（numpy 刚体 + 平面桌面 penalty 接触；球近似几何）。  
状态 \(s=(p,R,v,\omega)\)；\(\omega\) 在物体系。**禁止**把 `visual_logo` 写入 \(d(\cdot)\)。

\(g\cdot s=(p,Rg,v,g^\top\omega)\)。\(T_g\)：体系统量/接触点 \(x_O\mapsto g^\top x_O\)；世界系质心力不变。

## 对象（可证伪）

| ID | 物体 | \(G^*\)（候选上） | 用途 |
|----|------|-------------------|------|
| C0 | 无柄轴对称杯（圆柱） | \(\{R_y(\theta)\}\cup\{R_x(180^\circ),R_z(180^\circ)\}\) | 正例；候选格上的 \(D_{\infty h}\) |
| C1 | 有柄杯（偏置质量+几何） | \(\{I\}\) | 防止全对称 |
| C2 | 长方体 \(a\neq b\neq c\) | Klein：\(\{I,R_x,R_y,R_z\}(180^\circ)\) | 有限群 |
| C3 | 球 | 全部候选 | 连续 \(SO(3)\) |
| C4 | 同 C0 几何，\(I_x\neq I_z\) | \(\{I,R_y180,R_x180,R_z180\}\) | **拒 yaw quotient** |
| C5 | 同 C0 动力学 + 纯视觉 logo | 同 C0 | appearance 不得进入 \(D_H\) |

C4 的 \(R_y(90^\circ)\) **不得**接受。C5 的 logo 只记日志，不进 \(d\)。

## 候选（算法不知 \(G^*\)）

物体系三轴，\(\theta\in\{0^\circ,15^\circ,\ldots,345^\circ\}\)（\(0^\circ=I\) 只计一次）。  
本格 **不做** 反演/镜像（\(O(3)\setminus SO(3)\) 不是刚体位形）。

## Probe family（必须有激励）

每 seed **96** 条 \((s,a)\)，覆盖：

- 空中自由运动；
- 桌上世界系推（\(x/y\)）；
- 物体系侧面冲量；
- 物体系 \(\tau_x,\tau_y\)；
- 随机体素接触冲量；
- 初值自旋。

\(H=40\)，\(\Delta t=5\,\mathrm{ms}\)（0.20 s）。  
G-excite：至少 50% probe 的 \(d(s_H,s_0)>0.05\)。

## 距离与接受

无量纲

\[
d=\frac{\|\Delta p\|}{0.05}+\frac{\angle(R_1,R_2)}{\pi}+\frac{\|\Delta v\|}{0.3}+\frac{\|\Delta\omega\|}{5}.
\]

\(D_H\)：probe 上对时间平均再期望。

**接受**：\(D_H(g)\le\tau=0.04\)（含 \(I\) 的数值地板）。  
漏掉对称只是多存状态；**错误删除因果自由度更危险** ⇒ false quotient 门严格。

## Gates

| Gate | 条件 |
|------|------|
| G0-instrument | 所有物体 \(D_H(I)<0.01\)；G-excite 过 |
| G1-false-quotient | \(P(\mathrm{accept}\mid g\notin G^*)<0.01\)（C3 不计入分母） |
| G2-recall | \(P(\mathrm{accept}\mid g\in G^*)\ge0.90\)（C1 的 \(G^*=\{I\}\) 须接受 \(I\)） |
| G3-C4 | 不接受任何 \(R_y(\theta)\) 且 \(\theta\notin\{0,180\}\) |
| G4-C5 | C5 与 C0 对 \(R_y(\cdot)\) 的 accept 集相同 |

必报：每物体 AUROC(\(-D_H\), \(g\in G^*\))；C0 上 \(D_H(R_y)\) vs \(D_H(R_x(90))\)。

## Patterns

| Pattern | 条件 |
|--------|------|
| `instrument_failure` | ¬G0 |
| `shape_symmetry_confound` | G0 ∧ ¬G3（把几何对称当成因果对称） |
| `appearance_confound` | G0 ∧ ¬G4 |
| `causal_symmetry_supported` | G0–G4 |
| `symmetry_discovery_insufficient` | G0 ∧ ¬(G1∧G2) 且 G3∧G4 |

## 解锁

`causal_symmetry_supported` → 允许预注册 **SYM-X1**（已写 [`SYMX1_PREREG.md`](SYMX1_PREREG.md)，未开跑）。  
否则 X1/X2/X3 LOCKED。不得据此解锁 O0G6R / O1。
