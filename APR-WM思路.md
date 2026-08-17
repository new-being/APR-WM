到现在，项目已经从最初的 **“Physics + Residual 自适应计算”**，逐步演化成一个更完整的 **“能够发现自身物理解释不足、主动诊断、提出新结构并在物理约束下修正自己的世界模型”**。

我会把整个思路概括成一句话：

> **先用最简单的显式物理解释世界；只有当证据表明现有物理无法解释观测时，才调用 residual、主动获取信息或修改物理结构；任何新结构都必须经过物理可行性与预测效用验证后，才能写回 persistent world model。**

现在比较合适的项目概念已经不是单纯 APR-WM，而更接近我们前面讨论的 **PARSE-WM：Physics-Admissible Residual-guided Self-revising Embodied World Model**。

------

# 一、最开始的出发点：不要让神经网络什么都学

最初的问题其实很简单。

现实世界里有大量动力学可以被低复杂度物理规律解释，例如：

- 刚体运动；
- 弹簧；
- 阻尼；
- 重力；
- 简单碰撞。

如果全部交给高容量 neural world model 学，相当于浪费：

- 数据；
- 参数；
- 计算；
- 可解释性。

所以最初假设是：

\[ \boxed{ F_{\mathrm{world}} = F_{\mathrm{physics}} + gF_{\mathrm{residual}} } \]

其中 \(g\) 决定什么时候调用 neural residual。

**Residual，残差模型**：只学习显式 physics 没有解释掉的部分。

可复制 LaTeX：

```
$$
F_{\mathrm{world}}=F_{\mathrm{physics}}+gF_{\mathrm{residual}}
$$
```

核心直觉：

> physics 负责简单且结构化的规律，residual 只处理复杂部分。

------

# 二、V0：复杂计算真的可以稀疏分配吗？

## 假设

不是所有 interaction 都同样复杂，因此：

\[ \boxed{ \text{predictive complexity is sparse at interaction level} } \]

中文：

**预测复杂度在对象交互层级具有稀疏性。**

也就是说，有些 object pair：

\[ (i,j) \]

只靠 physics 就够了，有些才需要 residual。

------

## 实验

构造合成 2D object-interaction simulator，比较：

- Pure Physics；
- Pure Neural GNN；
- Always-on Residual；
- Adaptive Router。

初始核心结果中 Adaptive：

- OOD rollout RMSE ≈ `0.03175`；
- Always-on ≈ `0.03155`；
- router AUROC `0.962`；
- learned-compute proxy 约 `11%`。

------

## 得到的结论

第一层假设成立：

\[ \boxed{ \text{不是所有 interaction 都需要 neural residual} } \]

也就是说可以做：

**adaptive computation，适应性计算分配。**

但随后出现了更深的问题：

> physics 为什么会预测错？

------

# 三、V1：误差不是简单的“参数错 + 结构错”

最开始设想：

\[ e=e_{\mathrm{param}}+e_{\mathrm{structural}} \]

后来发现不成立。

正确的 factorial decomposition 是：

\[ \boxed{ I=E_{11}-E_{10}-E_{01}+E_{00} } \]

其中：

- parameter error：参数误差；
- structural error：结构误差；
- interaction term：二者的交互误差。

可复制：

```
$$
I=E_{11}-E_{10}-E_{01}+E_{00}
$$
```

------

## 关键现象：Compensation Trap

中文：

**误差补偿陷阱。**

错误的物理模型可以调整错误参数，让两个错误互相抵消。

例如真实模型有 nonlinear term，但模型只有 linear spring：

\[ F=-kx-cv \]

它可能学出一个错误：

\[ \hat k\neq k^* \]

反而在训练区域预测得很好。

这叫：

**pseudo-true parameter，伪真参数**

即：

> 错误模型族中最能拟合数据的参数，而不是真实物理参数。

V1 中 interaction term 为显著负值：

\[ -0.22169 \]

说明这种补偿确实存在。

------

## 得到一个非常重要的新结论

\[ \boxed{ \text{decomposition fidelity} \neq \text{short-term predictive optimum} } \]

中文：

> **物理解释正确，不等于短期 RMSE 最低。**

可复制：

```
$$
\text{decomposition fidelity}\neq\text{short-term predictive optimum}
$$
```

这是整个项目第一次从“计算优化”进入“认识论问题”。

------

# 四、V2：如果参数和结构可以互相伪装，机器人能不能主动问问题？

这时 belief 从：

\[ p(\theta\mid D) \]

扩展成：

\[ \boxed{ p(\theta,M\mid D) } \]

其中：

- \(\theta\)：物理参数；
- \(M\)：模型类别 / model class。

可复制：

```
$$
p(\theta,M\mid D)
$$
```

------

## 引入 Information Gain

**IG = Information Gain，信息增益。**

不是为了完成任务，而是选择一个动作，使不同物理假设产生尽可能不同的结果。

例如：

\[ IG(a)=I((\theta,M);Y\mid D,a) \]

中文：

> **这个动作执行后，预计能减少多少关于参数和模型类别的不确定性？**

可复制：

```
$$
IG(a)=I((\theta,M);Y\mid D,a)
$$
```

------

## 结果

Joint-IG 相比只考虑参数的 Parameter-IG：

- Model AUROC：`0.9155 → 0.9632`
- intervention RMSE：`0.1949 → 0.1740`
- probe 数量还减少。

但是更重要的是：

**action-only oracle 已经接近 Joint-IG，而 model-class oracle 好得多。**

所以发现：

\[ \boxed{ \text{probe action selection was not the main bottleneck} } \]

真正的问题变成：

> **怎么知道当前 physics family 本身错了？**

------

# 五、V3：如何区分“参数没调好”和“方程本身错了”？

这是 parameter tangent space 思路。

当前 physics：

\[ f_\theta(s,a) \]

参数变化能造成的预测变化：

\[ J_\theta\Delta\theta \]

其中：

\[ J_\theta=\frac{\partial f_\theta}{\partial\theta} \]

这些方向组成：

**parameter tangent space，参数切空间**

\[ \mathcal T_\theta=\operatorname{col}(J_\theta) \]

可复制：

```
$$
\mathcal T_\theta=\operatorname{col}(J_\theta)
$$
```

然后把 residual：

\[ r \]

分成：

\[ r_\parallel=P_{\mathcal T}r \]

和：

\[ \boxed{ r_\perp=(I-P_{\mathcal T})r } \]

可复制：

```
$$
r_\perp=(I-P_{\mathcal T})r
$$
```

解释：

- \(r_\parallel\)：可以通过调参数解释；
- \(r_\perp\)：当前 parameterized physics family 调任何附近参数都解释不了。

所以 persistent \(r_\perp\) 就成为：

**structural inadequacy evidence，结构失配证据。**

------

## V3 验证结果

Structural AUROC：

\[ 0.7709\rightarrow0.9957 \]

adequate false revision：

\[ 65.44\%\rightarrow0.92\% \]

说明：

> 看 residual 大小远远不够；看 residual 是否落在参数可解释方向之外才更本质。

但又出现新问题。

系统可以非常准确地说：

> “当前模型错了。”

可是面对 unseen dynamics：

\[ 89.08\% \]

能正确判成 `unknown`，预测却仍然很差。

于是：

\[ \boxed{ \text{model inadequacy detection} \neq \text{model discovery} } \]

中文：

> **知道自己错，不等于知道正确答案。**

------

# 六、V4：知道模型错以后，怎么提出新的物理结构？

于是引入有限：

**operator library，结构算子库。**

例如：

\[ \{q^3,\ qv,\ |v|v,\ v^2,\ldots\} \]

用候选 operator 去解释：

\[ r_\perp \]

例如：

\[ r_\perp\approx\alpha\phi_j(s,a) \]

并考虑 complexity cost：

> 谁能以最小结构复杂度解释最多 orthogonal residual？

------

## 整体流程变成

```
persistent r_perp
      ↓
model inadequacy
      ↓
operator proposal
      ↓
top-k candidates
      ↓
diagnostic intervention
      ↓
candidate selection
      ↓
held-out acceptance
```

------

## V4 结果

Top-3 proposal recall：

\[ 98.99\% \]

说明：

> 正确结构几乎总能进入候选集。

但 exact recovery 只有约：

\[ 60\% \]

Proposal oracle 只增加约 `0.38` 个百分点。

所以发现：

\[ \boxed{ \text{proposal nearly solved; selection and acceptance become limiting} } \]

也就是：

> **难点已经不是“想到什么解释”，而是“如何可靠地区分几个都说得通的解释”。**

------

# 七、V5：有限噪声下怎么区分候选？

V5 把 candidate coverage 控制为 100%，专门研究：

**noisy model discrimination，噪声条件下的模型判别。**

比较：

- normalized disagreement；
- posterior-weighted probes；
- fixed multi-probe；
- sequential probing；
- action oracle。

------

## 关键结果

更多、更加合理的 probe 确实能提升 selection。

三次固定 probe 达到最高非 oracle recovery：

\[ 79.42\% \]

Sequential 方法：

\[ 2.212\rightarrow0.906 \]

大幅降低平均 probe，但 recovery 降低，说明它形成的是：

\[ \boxed{ \text{evidence-cost Pareto frontier} } \]

中文：

**证据质量与证据采集成本之间的帕累托前沿。**

------

## 更重要的新发现

高噪声下：

selection 仍有：

\[ 73.06\% \]

但正确选择后的 acceptance 只有：

\[ 30.53\% \]

于是瓶颈再次转移：

\[ \boxed{ \text{candidate discrimination} \rightarrow \text{validation power} } \]

------

# 八、V6：什么时候才应该真正修改 persistent physics？

V6 把 acceptance 从固定 RMSE threshold 改成：

**sequential evidence acquisition，序贯证据积累。**

同时发现 V5 的 acceptance collapse 有两个来源：

1. 真实证据确实不足；
2. 固定阈值没有扣掉 observation noise floor。

------

## Noise calibration 的重要反例

校准噪声后 recovery 提升，但 false revision 也明显升高。

说明：

\[ \boxed{ \text{higher statistical power} \neq \text{safe revision} } \]

所以 acceptance 不能只比较：

\[ M_{\mathrm{new}}>M_{\mathrm{old}} \]

还要比较：

\[ M_{\mathrm{new}}>M_{\mathrm{runner-up}} \]

即：

> 新模型不仅必须比旧模型好，还必须有证据证明自己比其他竞争解释好。

------

## V6 结论

Sequential BF-style 方法形成：

\[ \boxed{ \text{validation power} \leftrightarrow \text{false revision risk} \leftrightarrow \text{evidence cost} } \]

三者之间的 Pareto，而不是免费提升。

到这里 synthetic 主线基本闭环。

------

# 九、然后开始真实性扩张：为什么第一次 R0 会失败？

把冻结 V6 接到 headless SAPIEN hinge 后，第一次结果非常差：

- tangent AUROC 接近随机；
- C1 recovery 几乎消失；
- pure physics 甚至优于完整 V6。

最初看起来像：

> 机制离开 toy simulator 就失效。

但后来发现实际原因不是这样。

------

# 十、R0.1：发现了 Representation / Interface Misspecification

旧接口近似使用：

\[ [\tau,\dot q]\rightarrow\ddot q \]

但真实 articulated dynamics 至少依赖：

\[ q,\dot q,M(q),g(q),\ldots \]

因此输入本身不物理闭合。

而且进一步发现 SAPIEN 还隐藏保留：

\[ 0.05 \]

joint friction。

------

## 修正后的关键表示

把 residual 从 acceleration space 改到：

**generalized-force space，广义力空间**

\[ \boxed{ r_\tau = M(q)\ddot q+h(q,\dot q)-\tau_{\mathrm{known}} } \]

可复制：

```
$$
r_\tau=M(q)\ddot q+h(q,\dot q)-\tau_{\mathrm{known}}
$$
```

这一步极其关键。

含义：

> neural residual 不再重新学习整个动力学，而只解释显式刚体动力学解释不了的“额外广义力”。

修正后 C0 residual 回到约 \(10^{-7}\)，C1 recovery 恢复到 100%。

所以第一次 realism no-go 被证明是：

\[ \boxed{ \text{representation/interface misspecification} } \]

而不是机制失败。

------

# 十一、R0.2：完整 V6 在 force-space 中重新接回

结果：

- C0 false revision 0%；
- C1 trigger/proposal/selection/acceptance/recovery 全 100%；
- operator coefficient 精确恢复；
- assimilation ratio = 1；
- rollout gain 随 horizon 增大。

这里第一次真正实现：

\[ \boxed{ \text{neural residual evidence} \rightarrow \text{explicit structural assimilation} } \]

中文：

**把原本由 neural residual 解释的误差，转化回显式物理结构。**

------

# 十二、R0.3：Native dynamics 暴露第二个 realism 问题

在人工 clean C1 中很好，但面对 SAPIEN native discrepancy：

one-step force validation：

\[ +0.00492 \]

看起来有改善。

然而 H16：

\[ -2.91 \]

严重恶化。

于是：

\[ \boxed{ \text{local validation success} \neq \text{long-horizon dynamical utility} } \]

中文：

**局部拟合更好，不代表形成了更好的动力系统。**

revision 可能修改整个 vector field，反复积分后失稳。

这就是更高层的：

**revision compensation trap，模型修正补偿陷阱。**

------

# 十三、同时 C2 又发现：知道 unknown 也不够

C2 中系统能很好地说：

> 我不知道。

但 memoryless residual fallback 仍然很差。

为什么？

因为真实状态具有 history dependence：

\[ F_t=F(s_t,a_t,h_t) \]

而 fallback 只看到：

\[ F(s_t,a_t) \]

所以缺的不是网络容量，而是：

**state closure，状态闭合性。**

加入 history-aware fallback 后，H16：

\[ 0.04995\rightarrow0.00243 \]

远优于 physics。

因此又得到：

\[ \boxed{ \text{epistemic adequacy} \neq \text{representational adequacy} } \]

中文：

> **知道自己不知道，不等于当前状态表示足够解决这个未知。**

------

# 十四、R0.4–R0.5：为什么短期安全的 revision 长期仍会炸？

R0.4 发现：

\[ H2/H4/H8\ \text{safe} \not\Rightarrow H16\ \text{safe} \]

于是开始找长期不稳定的动力学原因。

R0.5 diagnostic benchmark 发现最强信号之一是：

- positive revision power；
- negative effective damping；
- Jacobian spectral radius。

尤其 `abs_v_v` 正系数：

\[ r_\tau=\alpha|v|v,\qquad\alpha>0 \]

意味着：

\[ r_\tau v>0 \]

residual force 与速度同向。

也就是：

**anti-damping，反阻尼**

不断向系统注入能量。

------

# 十五、R0.5P：去掉 operator identity 混杂后，Passivity 成立

固定同一个 `abs_v_v` operator，只对称改变：

\[ \alpha=\pm\{0.15,0.30,0.45\} \]

结果：

- active：约 88–96% H16 instability；
- dissipative：0% instability。

而且 short-safe 的情况下依然如此。

于是建立了非常强的机制证据：

\[ \boxed{ \text{zero-input dissipativity} \Rightarrow \text{strong long-horizon stability prior} } \]

中文：

**零输入耗散性是长期稳定性的强物理先验。**

形式为：

\[ r_\tau(q,\dot q,0)\dot q\le0 \]

可复制：

```
$$
r_\tau(q,\dot q,0)\dot q\le0
$$
```

但同时发现：

\[ \boxed{ \text{dissipative} \not\Rightarrow \text{useful} } \]

因为过度阻尼虽然稳定，也可能预测很差。

------

# 十六、R0.6：最终形成“物理可行性 → 预测效用”的两层 revision

于是 revision 不再是：

> validation loss 小 → 接受。

而变成：

```
candidate revision
        ↓
Physical admissibility
        ↓
Predictive utility
        ↓
Accept / Reject
```

也就是：

\[ \boxed{ \text{physics decides what is allowed; data decides what is useful} } \]

中文：

> **物理先验决定什么不能做，数据决定剩下的候选中什么值得做。**

------

## Confirmatory 结果

在全新 seeds `7001–7041`：

Native H16：

\[ 0.32833\rightarrow0.23468 \]

改善：

\[ 28.52\% \]

同时：

- stability 100%；
- acceptance 85%；
- useful recall 93.68%；
- useful precision 87.25%；
- C0 false revision 0%；
- C1 exact recovery 100%。

所以目前最强的受支持结论是：

\[ \boxed{ \text{physical admissibility} + \text{predictive evidence} \Rightarrow \text{stable, nontrivial, aggregate-useful revision} } \]

可复制：

```
$$
\text{physical admissibility}+\text{predictive evidence}\Rightarrow\text{stable, nontrivial, aggregate-useful revision}
$$
```

但仍然**不能**说：

> 每一个 accepted revision 都有正长期 utility。

这是当前明确保留的限制。

------

# 十七、现在项目的整体机制已经变成什么？

从最初：

```
Physics
   +
Residual
```

已经发展成：

```
Current physical hypothesis
          ↓
       Prediction
          ↓
       Residual
          ↓
Parameter-explainable ?
    ↙              ↘
 yes               no
 ↓                  ↓
update θ       structural evidence
                      ↓
               model inadequate?
                      ↓
               operator proposal
                      ↓
              candidate hypotheses
                      ↓
             diagnostic evidence
                      ↓
            physical admissibility
                      ↓
             predictive validation
                      ↓
             accepted revision
                      ↓
          updated physical hypothesis
```

这就是一个：

\[ \boxed{ \text{self-revising physical world model} } \]

中文：

**自我修正物理世界模型。**

------

# 十八、Residual 的角色也已经发生了根本变化

最开始 residual 只有一个作用：

### 1. Prediction correction

中文：**预测修正**

后来增加：

### 2. Model inadequacy evidence

中文：**模型失配证据**

再后来：

### 3. Temporary fallback

中文：**新结构尚未发现时的临时预测兜底**

最终：

### 4. Structure discovery signal

中文：**驱动物理结构发现的信号**

所以现在最核心的想法之一可以写成：

\[ \boxed{ \text{Residual is not merely an error corrector; it is evidence about how the model is wrong.} } \]

中文：

> **残差不仅告诉模型“预测错了多少”，还告诉模型“可能错在哪里”。**

------

# 十九、到目前真正验证过什么，没验证什么？

这是必须严格区分的。

## 已有较强证据支持

### 1. Interaction-level adaptive residual allocation 可行

V0。

### 2. 参数误差和结构误差存在补偿

V1。

### 3. 主动信息动作有助于 model-class inference

V2。

### 4. Parameter-tangent orthogonal residual 能有效检测结构失配

V3。

### 5. 有限 operator library 可以高 recall 地提出缺失结构

V4。

### 6. 有限噪声下候选判别存在 evidence-cost tradeoff

V5。

### 7. Acceptance 是 validation power / false revision / evidence cost 的权衡

V6。

### 8. Physics-consistent force-space 表示是 realism transfer 的必要条件

R0.1。

### 9. Unknown fallback 需要状态闭合/history

R0.4-B。

### 10. Passivity 对长期稳定性是强机制性先验

R0.5P。

### 11. Passivity + 非平凡 acceptance 可以产生 aggregate-useful native revision

R0.6。

------

# 二十、仍然没有验证的东西

这也很重要。

目前**尚未证明**：

- RGB / depth 下也成立；
- object slots 下 tangent reasoning 仍可靠；
- 大规模 manipulation task 下成立；
- 多接触 / hybrid dynamics 下成立；
- real robot 上成立；
- arbitrary operator discovery 成立；
- 无限开放世界模型 discovery 成立；
- 每个 accepted revision 都长期有用；
- repeated revision / rollback 的 V7 问题已经解决；
- Diffusion action prior 已和当前 self-revision 系统整合。

尤其最初设计里的：

\[ q_\phi(A\mid G,l,m) \]

**成功经验动作 prior 目前仍主要属于整体架构规划，而不是已经验证的主实验链。**

这一点以后论文一定要分开。

------

# 二十一、现在正在做什么？

当前阶段：

**R1-MJ/RS：Frozen R0.6 under MuJoCo dynamics, robot contact, and task diversity**

R1-MS0（ManiSkill Drawer）保留为：

\[ \boxed{ \text{infrastructure\_block},\ \text{scientific\_no\_go}=\text{false} } \]

已完成的 realism bridge：

1. **R1-MJ0**：原生 MuJoCo 单 hinge，9-cell force-space closure 通过。
2. **R1-RS0**：robosuite `Door` Mode-A C0，9-cell 全部单独闭合。
3. **R1-RS1**：frozen R0.6 Door 科学迁移正式跑完 → `RS1_GO=false`，但结论是
   **差分可迁移性**，不是整体否定 R0.6（见 `REPORT/REP/R1_RS1_REPORT.md`）。

当前阶段链路：

\[
\boxed{RS1A\text{ (frozen)} \rightarrow \cdots \rightarrow V7A \checkmark \rightarrow V7B\text{ memory }\checkmark/\text{C0 }\times \rightarrow V7B.1/B.2 \times \rightarrow V7B.3\text{ evidence-warranted }\checkmark}
\]

**R1-RS1A** 整段已冻结（`REPORT/REP/R1_RS1A_STAGE_FREEZE.md`）：  
tolerate / probe / revise_worthy + VoI 最优激励 \(A^\star=1.5A_0\)。

**R1-RS1B** 正式 30-episode 已跑完（`REPORT/REP/R1_RS1B_REPORT.md`）：`RS1B_GO=false`。
\(\mathcal P_{\mathrm{rev}}\) 上 passivity 与 H32 utility 通过；binary modeled-support
retention 过严。

**R1-RS1B.1**（`REPORT/REP/R1_RS1B1_REPORT.md`）：`RS1B.1_GO=false`。
\(I_{\mathrm{exit}}>0\not\Rightarrow\mathrm{harmful}\)；连续 \(D\) 未胜过二元 \(I\)，因中间带为空。

**R1-RS1B.2** 正式矩阵已跑完（`REPORT/REP/R1_RS1B2_REPORT.md`）：`RS1B.2_GO=true`。
targeted query 填满 B0–B3（各 60 条，intended=realized）。中间带**可以存在**；
出域后 Spearman\((D,\mathrm{RMSE})\approx0.018\)，B2≈B3，主导的是 \(D=0\) 跨越而非 \(D\) 上的连续斜率。
**support-risk 分支在此收束**：不设 RS1B.3，不把 \(I_{\mathrm{exit}}>0\) 做成 hard veto。冻结句见 `REPORT/REP/R1_RS1B_STAGE_FREEZE.md`：support 是 confidence-state，不是 accept/reject safety。

**R1-RS1C** 正式 100-episode 已跑完（`REPORT/REP/R1_RS1C_REPORT.md`）：`RS1C_GO=true`。
分层政策相对 \(\pi_{\mathrm{none}}\) 与 support-veto 均提高 seed-mean H32 gain；
\(I_{\mathrm{exit}}\) 只抬 monitoring，不 veto。**不改写** `RS1B_GO=false`。
**R1-RS2-C0** 已通过（`REPORT/REP/R1_RS2_C0_REPORT.md`）：27/27 cells
闭合，contact \(J_c^\top f_c\) 与 contact-`efc` 双路一致，false revision
为 0；exposure adapter 冻结为 \(s^\star=2.0\)。
**R1-RS2 Formal** 已跑完（`REPORT/REP/R1_RS2_REPORT.md`）：`RS2_GO=false`。
C0 特异性与 tolerate 安全通过，因此失败不是 “contact 把 \(J^\top f\) 算进
structural residual”。失败在 allocation：contact \(D_0\) 达不到冻结的
Mode-A 检测阈值，VoI extra 0 次 promote，\(\pi_{\mathrm{RS1C}}\) 0 install。
\(\pi_{\mathrm{always}}\) 的 seed-mean H32 为正，说明若绕过 detect/VoI，
接触下修正仍可有用。不改写 `RS1C_GO`；禁止事后改阈值或 adapter。
**R1-RS2A** 已跑完（`REPORT/REP/R1_RS2A_REPORT.md`）：`RS2A_GO=true`。
adapter 对上了 \(X_\phi\)，但 \(\kappa_\perp\) 下降；\(X_{\phi,\perp}\) 比
\(X_\phi\) 更能跨域预测 \(D_0\)。这是诊断，不改 RS2 决策。
**R1-RS2B** 已跑完（`REPORT/REP/R1_RS2B_REPORT.md`）：`RS2B_GO=true`。
Mode-A \(C\) 在 C2 上低于 \(C_{\mathrm{tol}}\)，但接触下 incumbent 对 latch 真值的 \(L\approx0.14\gg C_{\mathrm{tol}}\)；
同一 \(\Delta C\) 泛函在接触域仍 \(\approx0\)。不改 \(C_{\mathrm{tol}}\)，不重开 RS2A。

**R1-RS3A 已跑完**（`REPORT/REP/R1_RS3A_REPORT.md`）：`RS3A_GO=false`。
H1 在单一 \(\tau_S\) 下 C0 FPR 通过，但 H2 失败：\(S_\perp\) 的 Mode-A/contact
gap 比 \(D_0\) 更大。局部 H0 白化不是跨干预标尺。
**冻结原则**：local normalization \(\neq\) semantic normalization；
same decision at one threshold \(\not\Rightarrow\) same evidential scale。
禁止继续改 \(S_\perp\) 的 H0。**R1-RS3A.1 已跑完**（`REPORT/REP/R1_RS3A1_REPORT.md`）：
`RS3A.1_GO=false`。H1a 过（C0 上超额风险≈0，不是 memorizer）；H1b 与 H2 失败，
\(g(E_{\mathrm{XR}})>g(D_0)\)。无条件标量证据标尺在这一表示层级上应视为不存在。
不开 RS3A.2 / RS3B / RS3C。

------

# 二十二、把整个发展过程压缩成一张表

| 阶段   | 最初问题                       | 发现/验证                                   | 新暴露的问题                                |
| ------ | ------------------------------ | ------------------------------------------- | ------------------------------------------- |
| V0     | 哪里需要 neural residual？     | residual compute 可稀疏分配                 | physics 为什么错？                          |
| V1     | 参数错还是结构错？             | 二者会产生 compensation                     | 如何主动区分？                              |
| V2     | 什么动作最有诊断价值？         | Joint-IG 有效                               | model-class inference 成瓶颈                |
| V3     | 当前模型族是不是错了？         | tangent evidence 很强                       | 知道错 ≠ 知道怎么改                         |
| V4     | 缺什么结构？                   | proposal recall ≈99%                        | selection/acceptance 成瓶颈                 |
| V5     | 噪声下如何选候选？             | multi-probe 有效，存在 evidence-cost Pareto | validation power 成瓶颈                     |
| V6     | 什么时候接受新结构？           | sequential validation 更省证据              | short/local validation ≠ long-term dynamics |
| R0.1   | realism 为什么失败？           | 找到 physics-interface mismatch             | 必须 force-space closure                    |
| R0.3   | local fit 能否保证 rollout？   | 不能                                        | vector-field stability                      |
| R0.4-B | unknown 后 residual 为什么差？ | 缺 history/state closure                    | fallback 必须带记忆                         |
| R0.5P  | 什么决定长期稳定？             | passivity 是强稳定先验                      | 稳定 ≠ 有用                                 |
| R0.6   | 如何安全而非平凡地 revision？  | admissibility + utility 总体有效            | 长期 utility calibration 尚未完美           |
| R1-MS0 | 能否进入 ManiSkill Drawer？    | plumbing 就绪，官方资产 TLS 阻塞            | 记为 infrastructure_block，非 scientific no-go |
| R1-MJ0 | MuJoCo force interface 是否闭合？ | 9/9 hinge cells 通过                     | robosuite Door asset 是否仍闭合             |
| R1-RS0 | Door Mode-A C0 是否跨参数闭合？ | 9/9 profiles×seeds 通过                     | frozen R0.6 能否迁移到 Door misspecification |
| R1-RS1 | revision 原则能否在 Door 上成立？ | **差分迁移**：safety/proposal 强；weak detection / H32 弱 | 拆成 RS1A（trigger）与 RS1B（stability） |
| R1-RS1A | inadequacy 是否需要 multi-source evidence？ | C2：support coupling 必需；persistence 有害 | 证据如何组合？ |
| R1-RS1A.1 | support 能否在不伤 C1 下救 C2？ | C2 可改善；**C1-L ≠ support-coupling** | 转向 weak-signal detector |
| R1-RS1A.2 | C1-L 信号在哪个 statistic？ | \(D_0\) 最强；coherence/match 不胜 \(D_0\)；typed OR 覆盖 C2 | excitation-limited? |
| R1-RS1A.3 | C1-L miss 是否 excitation-limited？ | **是**：recall 随 \(A\)/\(X_\phi\) 上升 | RS1A.4 |
| R1-RS1A.4 | 何时值得花 epistemic effort？ | tolerate/probe/revise 三分区成立；GO 因 miss 中 probe 占比较高未过 | VoI of probe |
| R1-RS1A.5 | probe 的 excitation 是否值得代价？ | **是**：\(A^\star=1.5A_0\)，\(V>0\)；\(2A_0\) flip=1 但 \(V<0\) | **RS1A 阶段冻结** |
| R1-RS1B | revise-worthy 的长期动力学是否可接纳？ | **GO=false**：utility/passivity 过；stable rate 0.80；support-exit 残差 | 禁止把 exit 做成 hard filter |
| R1-RS1B.1 | 离开 modeled support 对预测风险意味着什么？ | **GO=false**：exit \(\not\Rightarrow\) harm；连续未胜过二元 | 中间带为空 |
| R1-RS1B.2 | 中间 support 带能否被 targeted query 填上？ | **GO=true**：B0–B3 各 60；风险主要是 \(D=0\) 跨越 | **分支收束**；不设 RS1B.3；support 只作置信监测 |
| R1-RS1C | 分层政策（allocation + 物理可接受 + 短效用 + support 监测）能否在不 veto 下提升 H32？ | **GO=true**：30 install；vs none \(+0.00160\)；vs veto \(\Delta=+0.00108\) | **不改写** `RS1B_GO`；RS2 解锁 |
| R1-RS2 | 冻结 RS1C 在 robot-contact excitation 下是否仍安全且有用？ | **C0 PASS；Formal GO=false** | physics 可迁移，认识语义依赖上下文；禁止 contact 阈值重拟合 |
| R1-RS2A | 匹配的 \(X_\phi\) 为何不能保住 \(D_0\)？ | **GO=true**（诊断）：\(X_{\phi,\perp}\) 优于 \(X_\phi\) | 不改阈值；detectability 分支收束 |
| R1-RS2B | Mode-A \(C\) 在接触下是否仍是同一决策量？ | **GO=true**（诊断）：C2 的 \(\Delta C\approx0\) 但 \(L_{contact}\approx0.14\) | 不改 \(C_{tol}\)；不把 Formal 救活 |
| R1-RS3A | learner-visible 证据语义能否跨干预校准？ | **GO=false**：H1 过、H2 败；\(S_\perp\) gap \(>D_0\) | 禁止修 local null；只开 RS3A.1 |
| R1-RS3A.1 | held-out 超额结构风险能否跨干预校准？ | **GO=false**：H1a 过；H1b/H2 败 | **关闭** invariant scalar；转向 RS4A |
| R1-RS4A | 交互几何条件化证据能否 LOIO 校准？ | **GO=false**：H1 2/3；H2 \(0/3\) Brier 全面差于 \(D_0\) | 放弃 zero-shot transport；RS5A 仅为旁路 |
| R1-RS5A | 索引校准 + 支持域外弃权能否避免虚假自信？ | **GO=true**：indexed 优于 global；unseen FCR \(0/30\) | **不是主线**；不开 RS5B；transport 分支冻结 |
| R3-V7A | 混合具身域上情境化 \(p_{\mathrm{struct}}\) 能否域内校准？ | **GO=true**：Brier \(0.164<0.209\)；C0 FPR \(0.083\)；三族 AUROC \(\ge0.75\) | 不是 LOIO；不开 RS5B |
| R3-V7B | 持续 \(p_t\) 是否优于 V7A 静态 \(\tilde h\)？ | **GO=false**：H1/H2 过（终局 \(0.128<0.205\)，中点 \(0.129<0.303\)）；H3 C0 step FPR \(0.281>0.20\) | 记忆改善校准；失败是瞬态进入而非长期 hysteresis |
| R3-V7B.1 | 同一 GRU 上 IBS 能否压瞬态假信念且保留累积？ | **GO=false**：H1 IBS \(0.143>0.134\)；H2 FPR \(0.341\)；H3 非劣通过 | 不是 objective 就能修好 |
| R3-V7B.2 | 快慢状态+持续性门控能否降 C0 占用且保留累积？ | **GO=false**：H1/H3 过；H2 FPR \(0.595\)，\(B_{C0}\) 差于 B2；H4 差于 wide GRU | 时间尺度分离未解决占用 |
| R3-V7B.3 | 证据充分性监督能否降 C0 占用并跟踪 \(w_t\)？ | **GO=true**：H2 FPR \(0.006\)，\(B_{C0}\) \(0.219<0.293\)；H3 \(\rho=0.21\)；H4 配对间隙随证据增大；H1 非劣 | 同一 GRU；runtime 不见反事实 |
| R3-V7C | oracle 接触换成带噪本体感觉后，warranted \(p_t\) 还校准吗？ | **GO=true**：H2 FPR \(0.013\)；H3 \(\rho=0.24\)；H4 经 \(B_{C0}\) 与 C1-final 优于 N，混合 Brier 则 \(N<S\) | 不依赖 \(J^\top f\) |
| R3-V7C.1 | 学习触觉表示后 warranted 语义是否保持？ | **GO=false**：H1–H3 过；H4 败（\(B_{C0}\) 差于无接触）；taxel–oracle Spearman \(\approx 0\) | 不回头改 GRU；不开 V7D |
| R3-V7C.2 | 触觉信息在场 / encoder / fusion 何处丢失？ | **locus=FIELD**：\(\Delta_X=0.001\le\varepsilon\)；\(\Delta_Z<0\)；shuffle \(\approx0\) | 不要换 encoder；下一改动须是新观测定义 |
| R3-V7C.3 | 何种触觉观测对 \(e_t\) 相对 \(h^{-c}\) 有增量？ | **stop=false**：\(\Delta_{\mathrm{NSG}}=0.057\)；shear 嵌套未增益；\(N\) 相对 C.2 种子不稳定 | 不训 CNN / 不开 V7D；下一层仅 NSG 的 representation prereg |
| R3-V7C.4 | \(Z_{NSG}\) 能否保留 \(X_{NSG}\) 的条件增量？ | **SUFFICIENT**：同 run \(\Delta_Z\approx\Delta_X=0.015\)；shuffle 几乎不降 | 不训 B5 / 不开 V7D；LOSES 带在此 split 为空 |
| R3-V7C.5 | \(Z_{NSG}\) 接入 B5-S 能否改善 \(p_t\)？ | **GO=false**：H1/H4/H5 败；\(\Delta_{\mathrm{real}}=-0.046\)；shuffle 同样更差 | 表示充分性仍 PASS；认识有用性 FAIL；不开 V7C.6 |
| R3-V7D | RGB 相对 \(h^{S}\) 是否有条件增量？ | **GO=false**：H1/H4/H5 败；\(L(S)=0.441\)，\(L(S{+}Z)=0.488\)；shuffle 更好于 real | 与 C.5 同类；Door V7 关闭；无 V7E |
| R4-I0 | 宏观匹配且局部接触可分？ | **v1 PASS=false**（\(\mu\to F_n\)）；**v2 PASS=true**（左右 `solref` 交换，\(N_{\mathrm{matched}}=475\)） | I0 协议冻结；I1 已过 |
| R4-I1 | 完整 \(h^{S}\) 下 A/B 是否仍可辨？触觉增益？ | **PASS=true**：held \(\mathrm{AUROC}(h^{S})=0.556\)；增益 \(0.440\)；canonicalize \(\approx0.50\) | 轨迹划分；不改 I0；C0 已跑 |
| R4-C0 | 新接触域上 \(I(X_{\mathrm{tactile}};e_t\mid h^{S})>0\)？ | **GO=false**：\(\Delta_X=-0.015\)；\(w_t\equiv0\) | 基线 \(h^{S}\)；无 encoder；I0/I1 冻结 |
| R4-C1 | 局部差异是否通过**未来宏观后果**进入 \(e_t\)？ | **F0=true**；F1 二值 \(e\) 退化未评估；**GO=false** | 非 nuisance；不改 \(c_{\min}\)；无 encoder |
| R4-C2 | \(I(X;C_{\mathrm{future}}\mid h^{S})>0\)？ | **GO=false**：\(\Delta_C\approx 0\)；\(L\approx 1.055\) | **family STOP**；无 C3 |
| R5-I0 | 接触族三门能否共存？ | **PAUSE**：v1 \(C\)✓\(X\)×；v2 \(h^S\)×；v3 \(h^S\)✓\(X\)✓\(C\)× | 不是不可能定理；无 neural；再开须换 family |
| R5-I0-SELFSTRESS | 隐藏预应力→几何刚度能否三门+range+\(X{\leftrightarrow}C\)？ | **PASS=false**：\(D_h=0\)，\(X=\lambda\)；G_cons/G_xc Spearman \(0.75\) | 无 neural；判决冻结；不改 \(C\) |
| R5-I1-SELFSTRESS | \(I(X;Y^{\mathrm{future}}\mid h^{S})>0\)？（ridge \(\Delta_Y\)） | **GO=true**：\(\Delta_Y/L_0=0.387\)；shuffle 更差 | 不改 I0；无 encoder |
| R5-D0-preflight | oracle \(a^\star(\lambda)\) 是否 crossover？ | **PASS=true**：\(\lambda=0\to\) cons，\(\lambda\ge2\to\) mid | 未开 planner；不改 \(J\) |

------

# 二十三、现在这个项目最本质的思想

如果不谈具体版本，整个项目已经可以浓缩成四层。

### 第一层：Occam

> 能用简单 physics 解释，就不要调用复杂模型。

### 第二层：Epistemic diagnosis

> 预测错了以后，不要立刻加 neural capacity，先判断为什么错。

### 第三层：Scientific revision

> 如果现有 physics family 不够，就从 residual 中提出新结构，再主动验证。

### 第四层：Physical admissibility

> 数据拟合好不能成为物理模型修正的充分条件；新结构必须满足相应动力学约束。

### 第五层：Calibrated decision, not raw magnitude

> 世界模型不应该把“异常有多大”直接等价成“该不该采取行动”。异常、认识价值、物理可行性和预测后果必须分层决策。

### 第六层：Contextual / persistent epistemics

> 认识决策量的含义取决于具身历史 \(h_t\)（含交互方式），而不是一把跨域不变的标量。
> 最终系统维护 \(b_t=(s_t,\theta_t,M_t,u_t^{\mathrm{epi}})\)，把 intervention
> 当作历史中的因子，而不是手工 domain shift。
> \(u_t^{\mathrm{epi}}\) 应累计的是**足以正当化结构不充分性信念的证据**，
> 而不是原始异常幅度；V7C 表明这不要求精确 oracle 接触真值。
> 小域零样本运输失败（RS3–RS4）支持情境化；RS5A 只说明：若仍把域拆开，则需要索引校准与弃权。

### 第七层：Conditional modality value

> 一个模态在隔离条件下有信息，并不意味着它在当前 belief 下仍有认识价值。
> \(I(Z;e)>0\not\Rightarrow I(Z;e\mid h)>0\)。
> 世界模型不是把所有传感器都融合，而是只吸收相对已有历史具有条件增量的模态。

### 第八层：Conditional consequential information（R4）

> 能观测到潜变量 \(\neq\) 能观测到该潜变量中**决策相关**的变异。
> 不是“这个潜变量重要吗”，而是“当前能观测到的变化，是否在已有 belief 之外进一步解释了未来重要结果的变化”。
> 表示容量只应跟 **observable variation that carries conditional
> information about consequential variation**。
> R4 停在 \(I(X;C\mid h^{S})\le 0\)（A/B 可见，后果不被该观测解释）。
> R5 接触族 I0 v1–v3 把同一原则收成连续形式后 **PAUSE**（不是不可能定理）：
> \(I(X;\lambda\mid h^{S})>0\not\Rightarrow I(X;C\mid h^{S})>0\)。
> Self-stress I0：当前侧解析成立，但 \(C=\|Y-Y_0\|_2\) fold（`PASS=false`，冻结）。
> I1：不改该 \(C\)，直接问 \(I(X;Y^{\mathrm{future}}\mid h^{S})\)；ridge \(\Delta_Y/L_0=0.387\)，`GO=true`。
> D0-preflight：冻结任务 \(J\) 下 oracle \(a^\star\) 随 \(\lambda\) 翻转（`PASS=true`）；预测信息 \(\neq\) 决策信息仍待 regret 闭环。
> 边界：后果存在 \(\neq\) 到一条 baseline 的径向距离是合法全局坐标。
> 账本：`REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`，`REPORT/REP/R5_I1_SELFSTRESS_REPORT.md`，`REPORT/REP/R5_D0_PREFLIGHT_REPORT.md`。
> 无 encoder。Door 上触觉 C.5 与 RGB V7D 均为负：\(h^{S}\) 已饱和当前结构错误判断；继续加模态可以主动有害。
> 因此 **无 V7E**；R4 已证明相对 \(h^{S}\) 的观测必要性问题，不先训 encoder。

RS1 后半段已经形成对称：

\[
\boxed{
\begin{aligned}
RS1A &: \text{error magnitude}\not\Rightarrow\text{worth revising}\\
RS1A.5 &: \text{more excitation}\not\Rightarrow\text{more net epistemic value}\\
RS1B.1 &: \text{support exit}\not\Rightarrow\text{revision harmful}\\
RS1B.2 &: \text{larger support distance}\notRightarrow\text{proportionally larger risk}\\
RS2A &: \text{matched operator exposure}\notRightarrow\text{matched structural detectability}\\
RS2B &: \text{Mode-A consequence }C\notRightarrow\text{contact decision meaning}\\
RS3A &: \text{same }D_0\text{ magnitude}\notRightarrow\text{same evidential meaning across }\mathcal I\\
RS3A.1 &: \text{held-out excess structural risk}\not\Rightarrow\text{intervention-stable evidence}\\
RS4A &: \text{learner-visible }z_{\mathcal I}\not\Rightarrow\text{transported }p(Y\mid D_0,z)\\
RS5A &: \text{indexed }\mathcal C_{\mathcal I}+\text{abstention can work when domains are split}
\end{aligned}
}
\]

共同原则：

\[
\boxed{\text{raw diagnostic magnitude should not be used directly as a decision rule}}
\]

必须经过 consequence / value / confidence calibration。RS2–RS5 再加一条：

\[
\boxed{
\text{epistemic meaning is contextual; intervention is part of }h_t
}
\]

最终是 \(\pi_{\mathrm{epi}}(h_t)\) 与域内组合泛化，而不是
\(E_{\mathrm{ModeA}}\approx E_{\mathrm{contact}}\)。

因此现在最浓缩的一句话是：

\[ \boxed{ \text{predict} \rightarrow \text{diagnose} \rightarrow \text{hypothesize} \rightarrow \text{intervene} \rightarrow \text{validate} \rightarrow \text{revise} } \]

可复制：

```
$$
\text{predict}\rightarrow\text{diagnose}\rightarrow\text{hypothesize}\rightarrow\text{intervene}\rightarrow\text{validate}\rightarrow\text{revise}
$$
```

它已经很接近一个简化版的“机器科学推理循环”：

> **先有一个物理解释；用预测误差检验它；区分参数错误和结构错误；对结构错误提出新假设；通过干预获取证据；排除违反物理规律的解释；最后才修改世界模型。**

当前 realism 线上 RS1C 与 **RS2-C0** 已通过，**RS2 Formal 未通过**，
RS2A/B 是**诊断通过**而不是把 Formal 救活。阶段冻结见
`REPORT/REP/R1_RS2_STAGE_FREEZE.md`：

\[
\boxed{\text{The physics can transfer while the epistemics are context-dependent.}}
\]

Transport 主线冻结见 `REPORT/REP/R1_TRANSPORT_STAGE_FREEZE.md`。
不要重跑 RS2，不要 contact-specific \(\tau\) 重拟合，不要 RS5B。
**R1-RS5A** 作为旁路架构结果保留（`RS5A_GO=true`），不是主实验。
**R3-V7A** 已跑完（`REPORT/REP/R3_V7A_REPORT.md`）：`V7A_GO=true`。
混合域情境化 \(p_{\mathrm{struct}}\) 在 held-out seed 上校准；不是 LOIO
transport，也还不是持续 \(b_t\)。
**R3-V7B** 已跑完（`REPORT/REP/R3_V7B_REPORT.md`）：`V7B_GO=false`。
GRU \(p_t\) 终局与中点 Brier 均优于 V7A 静态摘要；失败是 C0 瞬态把结构信念
抬高，不是长期 hysteresis。
**R3-V7B.1** 已跑完（`REPORT/REP/R3_V7B1_REPORT.md`）：`V7B.1_GO=false`。
同一 GRU 改 IBS 不能修好瞬态占用。
**R3-V7B.2** 已跑完（`REPORT/REP/R3_V7B2_REPORT.md`）：`V7B.2_GO=false`。
快慢+门控保留并改善了对 B2 的累积，但不降低 C0 占用。
**R3-V7B.3** 已跑完（`REPORT/REP/R3_V7B3_REPORT.md`）：`V7B.3_GO=true`。
匹配 C0/C1 反事实上的 \(w_t\) 监督压低 C0 占用（FPR \(0.006\)）且 \(p_t\) 跟踪证据；
同一 GRU，runtime 不用反事实。
**R3-V7C** 已跑完（`REPORT/REP/R3_V7C_REPORT.md`）：`V7C_GO=true`。
推理时 oracle 接触换成带噪腕部/关节残差后，证据时序与 C0 占用仍在（FPR \(0.013\)）；
接触信息对认识决策有用，但不是所有汇总指标上都单调增益。
阶段冻结见 `REPORT/REP/R3_V7_STAGE_FREEZE.md`。
**R3-V7C.1** 已跑完（`REPORT/REP/R3_V7C1_REPORT.md`）：`V7C.1_GO=false`。
H1–H3 过、H4 败：触觉通道相对无接触没有认识增益。不要怀疑 V7B.3/V7C；
归因于触觉表示瓶颈。V7D / RGB 锁定。
**R3-V7C.2** 已跑完（`REPORT/REP/R3_V7C2_REPORT.md`）：locus **FIELD**。
相对 \(h^{-c}\)，8×8 taxel 对 \(e_t\) 无增量（\(\Delta_X=0.001\)）。不要换 CNN / 开 fusion / 开 V7D。
**R3-V7C.3** 已跑完（`REPORT/REP/R3_V7C3_REPORT.md`）：触觉分支 stop **未**触发。
最大增量是 \(X_{NSG}\)（\(\Delta=0.057\)）；剪切嵌套未证明增量；\(N\) 不推翻 C.2。
不开 encoder / V7D。
**R3-V7C.4** 已跑完（`REPORT/REP/R3_V7C4_REPORT.md`）：冻结判决 **SUFFICIENT**。
同 run \(\Delta_X=0.015\)、\(\Delta_Z=0.014\)；shuffle 几乎不变。不训 B5 / 不开 V7D。
**R3-V7C.5** 已跑完（`REPORT/REP/R3_V7C5_REPORT.md`）：`V7C.5_GO=false`。
在 B5-S 上追加 \(Z_{NSG}\) 使 \(e_t\) BCE 与 Brier 变差；时间门控失败。不开 V7C.6。
**R3-V7 Door 触觉支线已冻结**（`REPORT/REP/R3_V7_EVIDENCE_CHAIN.md`）：
表示充分 \(\neq\) 条件有用；本分布上不再扩 CNN/fusion。无 V7C.6。
**R3-V7D** 已跑完（`REPORT/REP/R3_V7D_REPORT.md`）：`V7D_GO=false`。
冻结 RGB encoder 相对 \(h^{S}\) 无条件增量（real 比 shuffle 更差）。
**Door / R3-V7 正式关闭**（`REPORT/REP/R3_V7_STAGE_FREEZE.md`）：无 V7E。
\(h^{S}\) 已饱和当前 Door 结构错误判断。
**R4-C0 已跑完**（`REPORT/REP/R4_C0_REPORT.md`）：`R4_C0_GO=false`。
\(\Delta_X=-0.015\le\varepsilon\)。宏观残差能量 \(\sim10^{-7}\)，\(w_t\equiv0\)。
I1 证明局部可观测；C0 证明该局部状态不进入当前 \(e_t\)。不训 encoder，不改 I0。
**R4-I0 v1 未过、v2 已过**（`REPORT/REP/R4_I0_REPORT.md`、`REPORT/REP/R4_I0_V2_REPORT.md`）。
v1：标量 \(\mu\) 泄漏进 \(F_n\)。v2：同 \(\mu\)、同平均刚度、左右 compliance 交换，宏观 \(h^{S}\) 对齐且 \(p,\tau_y\) 空间翻转。
**R4-I1 已过**（`REPORT/REP/R4_I1_REPORT.md`）：完整宏观通道、按 \(q_0\) 划分；
\(\mathrm{AUROC}(h^{S})=0.556\)，增益 \(0.440\)。C0 已跑且未过。
**R4-C1 已跑**（`REPORT/REP/R4_C1_REPORT.md`）：F0 过（弱未来分叉）；
F1 二值标签饱和未评估；`GO=false`。不是 nuisance。不训 encoder。
**R4-C2 已跑且停线**（`REPORT/REP/R4_C2_REPORT.md`）：`GO=false`。
连续 \(C\) 上 \(\Delta_C\approx 0\)。未来后果存在 \(\not\Rightarrow\) 触觉能量化它。
**R4 family 正式停止**（`REPORT/REP/R4_FAMILY_STOP.md`）。无 C3。
**R5-I0 接触族已 PAUSE**（`REPORT/REP/R5_CONTACT_FAMILY_PAUSE.md`）：
v1 \(C\)✓\(X\)×；v2 \(X\) 部分✓、\(h^{S}\)×、\(C\)×；v3 \(h^{S}\)✓\(X\)✓\(C\)×。
不是不可能定理。无 neural。不回到 R4。不再枚举接触表面 pattern。
**R5-I0-SELFSTRESS 已跑且判决冻结**（`REPORT/REP/R5_I0_SELFSTRESS_REPORT.md`）：
\(D_h=0\) 且 \(X=\lambda\)，标量 \(C=\|Y-Y_0\|_2\) fold，`PASS=false`。
**R5-I1-SELFSTRESS `GO=true`**（`REPORT/REP/R5_I1_SELFSTRESS_REPORT.md`）：
\(X\) 降低 held 未来宏观轨迹风险 \(\Delta_Y/L_0=0.387\)。不改 I0 \(C\)。无 encoder。
**R5-D0-preflight `PASS=true`**（`REPORT/REP/R5_D0_PREFLIGHT_REPORT.md`）：
oracle \(a^\star\) 在 \(\lambda=0\) 为 cons、其余为 mid。未测 regret。
更长程的 realism 问题仍然是：

\[ \boxed{ \text{这套循环能否从受控 SAPIEN 动力学迁移到真实多资产、多接触、多任务的 manipulation 环境？} } \]

这就是 robosuite / MuJoCo（以及后续 RoboTwin）等 realism expansion 真正要回答的问题。