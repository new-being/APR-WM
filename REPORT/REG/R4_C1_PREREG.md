# R4-C1 Preregistration — Consequence-Warranted Local Epistemic Value

Date: 2026-08-17  
Status: **FROZEN** (run; F0 true; F1 unevaluable; `R4_C1_GO=false`)  
Depends on: `REPORT/REP/R4_C0_REPORT.md` (`R4_C0_GO=false`),
I0 v2 PASS, I1 PASS  
Does not change: I0 v2 physics, I1 \(h^{S}\), R4-C0 question or GO,
Door V7, B.3 \(w_t\) recipe as a *closed* C0 object  
Locks: tactile encoder/CNN; I0 \(\mu\)/`solref`/grid/squeeze/\(F_t(t)\);
using \(X\) to build the label; weakening \(h^{S}\); patching C0

## Why this is a new stage, not a C0 patch

C0 asked \(I(X;e_t\mid h^{S})>0\) with B.3

\[
e_t=y\,w_t,\qquad w_t=\text{warrant(macro residual divergence)}.
\]

I0 v2 was built to be macro-equivalent and locally distinct, so that
\(w_t\) never armed (\(e_t\equiv0\)). That is a **target-domain**
failure, not a sensor/representation failure.

\[
\boxed{
\text{locally observable}
\neq
\text{warranted under a macro-residual evidence definition}
}
\]

C0 stays closed. This stage changes **what counts as worth knowing**.

\[
\boxed{
\textbf{R4-C1 — Consequence-Warranted Local Epistemic Value}
}
\]

中文名：**后果充分的局部认识价值**。

## Question

\[
\boxed{
\text{Can a locally hidden contact difference become epistemically
warranted through its future consequences before it appears in
current macro residuals?}
}
\]

Operationally, after a frozen diagnostic action \(a^{\mathrm{diag}}\):

\[
\boxed{
I\bigl(X_t;\,e_t^{\mathrm{future}}\mid h_t^{S}\bigr)>0\ ?
}
\]

This is the R4 mirror of RS1A.4:

\[
\text{detectability}\neq\text{consequence},
\qquad
\text{current macro residual}\neq\text{future consequence}.
\]

## Forbidden label

\[
e_t\neq f(X_t).
\]

Tactile is a **predictor input only**. The label is a simulator
counterfactual on **future macro** \(Y\).

## Frozen physics and sensors

I0 v2 / I1 unchanged: \(\mu=0.80\), mean `solref` matched, left/right
compliance swap, same squeeze and training \(F_t(t)\), same taxel
maps, over-complete \(h^{S}\) (no per-finger split forces).

## Frozen diagnostic probe (implementation lock)

On each matched A/B pair, at frozen time \(t_\star=\) last hold-phase
frame (\(t=0.20\,\mathrm{s}\), \(F_t^{\mathrm{cmd}}=0\), grasp already
on):

1. Snapshot \((q,\dot q)\) (macros should match; I0 Gate0).
2. Branch both worlds with the **same** extra tangential command
   \(a^{\mathrm{diag}}\): hold finger PD, set
   \(F_t^{\mathrm{cmd}}=F_t^{\mathrm{probe}}=4.8\) for
   \(H=0.40\,\mathrm{s}\) (200 steps at \(\mathrm{d}t=0.002\)).
   Do **not** retune \(\mu\) or `solref` to force a split.
3. Future macro (no taxels)

\[
Y_{t_\star:t_\star+H}
=
\{q,\dot q,F_n,F_t,F_{x,y,z},M_{x,y,z}\}.
\]

4. Consequence (label source)

\[
C_{t_\star}
=
\bigl\|Y^{A}-Y^{B}\bigr\|_{2}
\big/
\bigl(0.5\|Y^{A}\|_2+0.5\|Y^{B}\|_2+\varepsilon\bigr).
\]

Split by trajectory \(q_0\), never by random frames.

## Gate F0 — future consequence exists

\[
\texttt{R4\_C1\_F0}
\iff
\text{held-out mean }C_{t_\star}>c_{\min}=10^{-3}.
\]

If F0 fails:

\[
\boxed{
\text{local compliance swap is a nuisance state}
}
\]

Stop the R4 family. Do **not** train \(P_0/P_X\). Do **not** redesign
tactile. World models need not represent every physically real,
sensor-visible latent — only latents that change prediction or
decision.

中文：**世界里真实存在、传感器也能看到的变量，不代表世界模型就必须表示它；只有会改变预测或决策的变量才值得占用认识容量。**

## Gate F1 — conditional value for consequence-warranted \(e_t\)

Only if F0 passes. Define

\[
e_{t_\star}^{\mathrm{future}}
=
\mathbf{1}\{C_{t_\star}>c_{\min}\}.
\]

**Linear probe at \(t_\star\)** (no CNN, no GRU search; GRU is unused
because the query is a single prefix time):

\[
P_0:h^{S}_{t_\star}\to e^{\mathrm{future}},
\qquad
P_X:(h^{S}_{t_\star},X_{t_\star})\to e^{\mathrm{future}}.
\]

\(h^{S}_{t_\star}\) is the I1 over-complete macro history of width
\(W=8\) ending at \(t_\star\). Split by \(q_0\).

\[
\mathrm{AUROC}(h^{S})<0.90,
\qquad
\mathrm{AUROC}(h^{S},X)-\mathrm{AUROC}(h^{S})>0.10.
\]

**F1 GO** iff both hold on held-out \(q_0\). \(\mathrm{AUROC}(h^{S})\ge0.95\)
is leak-fail (future consequence already in macros).

Optional diagnostic (not GO): canonicalize/flip \(X\) as in I1; gain
should collapse if the predictor is using spatial allocation.

## Unlock / stop

| Outcome | Action |
|---|---|
| F0 fail | **R4 family stop**: nuisance latent; no encoder |
| F0 pass, F1 fail | Local state is future-consequential but not readable from \(X\) given \(h^{S}\) at \(t_\star\) — new prereg only, still no CNN shopping |
| F0 and F1 pass | Then, and only then, a **new** representation-sufficiency prereg for this \(X\) |

C0 remains `GO=false`. This stage does not revive B.3 \(w_t\) on
current residuals.

## What this does not authorize

- patching R4-C0 \(\varepsilon\) or \(w_t\)
- \(e_t=f(X)\)
- I0 retune to manufacture slip
- Door V7E, DINO/CLIP, RGB+tactile fusion, larger GRU
- treating I1 AUROC \(0.996\) as F1 GO
