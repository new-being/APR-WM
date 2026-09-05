# Transport Branch Freeze — Epistemics Are Context-Dependent

Date: 2026-08-16  
Status: **STAGE FROZEN**  
Does not change frozen GOs.  
Does **not** open RS5B, RS4A.1, RS3A.2, revision, VoI, or H32.

## One-line verdict (refined)

\[
\boxed{
\text{The physics can transfer while the epistemics are context-dependent.}
}
\]

中文：同一个世界的动力学表示可以跨交互机制迁移；认识量的含义取决于当前
具身历史，而不是一把到处通用的标量尺子。

这比早期的 “epistemics do not transfer” 更准确。负结果否定的是

\[
\text{cross-domain zero-shot transport of evidential meaning},
\]

不是 “永远无法学习认识状态”。

## Why the experimental split was artificial

Mode-A vs contact 是人为小域。真实机器人面对的是联合具身分布

\[
\mathcal D_{\mathrm{embodied}}
=
p(x^{\mathrm{vision}},x^{\mathrm{tactile}},q,\dot q,a,\mathrm{contact},\mathrm{task},\ldots).
\]

在这个大域上真正需要的是 **within-domain / compositional generalization**，
不是把一把尺子从 \(\mathcal I=A\) 零修改搬到 \(\mathcal I=B\)。

Intervention 应进入历史

\[
\mathcal I_t \subset h_t,
\]

由统一模型 \(\pi_{\mathrm{epi}}(h_t)\) 使用，而不是手工 domain router。

## Frozen chain (keep as mechanism results)

\[
\boxed{
\begin{aligned}
RS2\text{-C0} &: \text{contact }J^\top f\text{ accounting closes}\\
RS2\text{-Formal} &: \text{frozen RS1C allocation does not transfer}\\
RS2A &: \text{detectability is not raw-exposure invariant}\\
RS2B &: \text{Mode-A }C\text{ is not intervention-invariant}\\
RS3A/A.1 &: \text{no tested invariant scalar preserves evidential scale}\\
RS4A &: \text{geometry fingerprints }\neq\text{ transported calibrated }p\\
RS5A &: \text{indexed }\mathcal C_{\mathcal I}+\text{abstention works when domains are treated as separate}
\end{aligned}
}
\]

Shared scientific content:

\[
\boxed{\text{epistemic meaning is contextual}}
\]

## What RS5A is, and is not

`RS5A_GO=true` remains. It shows: if one *chooses* to treat interventions
as separate calibration objects, then

\[
(\mathcal I,\;\mathcal C_{\mathcal I},\;\mathrm{valid}(\mathcal C_{\mathcal I}))
\]

beats a global ruler and abstains outside support.

That is **not** the default next mainline for a unified embodied world
model. Continuing RS5B (typed \(C\), more routers, more small-domain
thresholds) would keep optimizing an artificial split.

RS5A’s abstention idea can later be absorbed as one head of a structured
epistemic belief (\(p_{\mathrm{unknown}}\), support / extrapolation).
It does not authorize more transport-calibration cells.

## Abandoned as a main requirement

\[
\boxed{
E_{\mathrm{ModeA}}\approx E_{\mathrm{contact}}
\quad\text{and}\quad
\text{zero-shot }p(Y\mid D_0)\text{ across unknown }\mathcal I
}
\]

Also abandoned as default next steps:

- RS3A.2 / smarter whitening / extra invariant scalars;
- RS4A.1 / more \(z\) features for LOIO transport;
- RS5B/C as the continuation of indexed small-domain calibration;
- contact-specific \(\tau\) / \(C_{\mathrm{tol}}\) refits of frozen RS1C.

## Redefined generalization

Need:

\[
\mathcal D_{\mathrm{train}}^{\mathrm{embodied}}
\rightarrow
\mathcal D_{\mathrm{test}}^{\mathrm{embodied}}
\]

new objects, masses, friction, contact points, actions, tasks, operator
combinations, cameras — still from the same embodied generative process
(**in-distribution compositional generalization**).

Still need uncertainty / abstention when

\[
\mathcal D_{\mathrm{deploy}}\not\subset\mathrm{support}(\mathcal D_{\mathrm{train}})
\]

(cloth, fluids, unseen mechanisms). That does **not** require a globally
identical scalar meaning.

## Next main stage

\[
\boxed{
\textbf{R3-V7A — Mixture-trained contextual epistemic belief}
}
\]

Prereg: `REPORT/REG/R3/R3_V7A_PREREG.md`.

Not RS5B. First step uses oracle state + proprioception + oracle contact
summaries of \(h_t\). Sequential \(b_t=(s_t,\theta_t,M_t,u_t^{\mathrm{epi}})\)
and vision/tactile come later, staged.

## Frozen GO status

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\text{-C0\_PASS} &= true\\
RS2\_GO &= false\\
RS2A/B\_GO &= true \quad\text{(diagnostic)}\\
RS3A/A.1\_GO &= false\\
RS4A\_GO &= false\\
RS5A\_GO &= true \quad\text{(side architecture; not mainline)}\\
RS3B/C,\ RS4B/C,\ RS5B/C &= locked\\
\text{transport mainline} &= \textbf{closed}
\end{aligned}
}
\]
