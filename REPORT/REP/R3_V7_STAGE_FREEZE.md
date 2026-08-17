# R3-V7 Stage Freeze — Evidence-Warranted Belief without Oracle Contact

Date: 2026-08-17  
Status: **FROZEN** (Door / R3-V7 **closed**; there is **no V7E**)  
Does not change: any RS* GO, C0, \(s^\star\), RS1C policy, Door V7 results  
Does not open: visuo-tactile fusion, RGB/tactile encoder search, DINO/CLIP,
larger GRU, modality routers, Door capacity shopping  
Next: **R4 STOP** (`REPORT/REP/R4_FAMILY_STOP.md`); R5 prereg not run

## Four layers (Door V7, complete)

\[
\boxed{
\begin{aligned}
1.\ \text{Context} &\quad\text{context helps}\quad\text{V7A }\checkmark\\
2.\ \text{Temporal belief} &\quad\text{history helps}\quad\text{V7B }\checkmark
  \text{ (needs correct temporal supervision)}\\
3.\ \text{Epistemic semantics} &\quad
  \text{supervise when evidence warrants belief,
  not merely whether the world is wrong}\quad\text{V7B.3 }\checkmark\\
4.\ \text{Conditional sensing} &\quad
  \text{a modality enters belief only if it adds information
  given what the agent already knows}\\
  &\quad\text{V7C }\checkmark;\ \text{Door tactile C.5 }\times;\ \text{Door RGB V7D }\times
\end{aligned}
}
\]

## Frozen chain

\[
\boxed{
\begin{aligned}
V7A &: \text{context helps calibration}\quad\checkmark\\
V7B &: \text{history helps calibration}\quad\checkmark\\
V7B.1 &: \text{objective-only fix insufficient}\quad\times\\
V7B.2 &: \text{fast/slow structure alone insufficient}\quad\times\\
V7B.3 &: \text{evidence-warranted supervision works}\quad\checkmark\\
V7C &: \text{oracle contact can be replaced by noisy proprioceptive contact sensing}\quad\checkmark\\
V7C.1 &: \text{learned tactile representation did not preserve a useful epistemic channel}\quad\times\\
V7C.2 &: \text{locus = FIELD (8\times 8 taxel not incremental for }e_t\text{ given }h^{-c}\text{)}\quad\mathrm{diagnostic}\\
V7C.3 &: \text{stop=false; }\Delta_{\mathrm{NSG}}=0.057;\ \text{geometry, not shear}\quad\mathrm{observation}\\
V7C.4 &: \texttt{SUFFICIENT}\ (\Delta_Z\approx\Delta_X\text{ same-run; shuffle-null)}\quad\mathrm{representation}\\
V7C.5 &: \text{belief integration of }Z_{NSG}\text{ vs B5-S}\quad\times\\
V7D &: I(Z_{\mathrm{RGB}};e_t\mid h^{S})\text{ vs }h^{S}\quad\times\\
V7E &: \text{does not exist}
\end{aligned}
}
\]

## Strongest positive result

\[
\boxed{
\text{Persistent contextual epistemic belief is feasible without oracle contact truth,
provided it is trained with evidence-warranted temporal supervision.}
}
\]

中文：**在统一具身混合分布上，只要监督的是“证据何时足以支持信念”，持续更新的认识状态可以在不依赖精确 oracle 接触真值的情况下保持校准。**

Also freeze:

\[
\boxed{
\text{the world being wrong}
\neq
\text{the current history warranting belief that it is wrong}
}
\]

\[
\boxed{
\text{warranted belief does not require perfect access to the causal variable}
}
\]

Independent design principle (new):

\[
\boxed{
\text{modality availability}\neq\text{conditional modality value}
}
\]

\[
\boxed{
I(Z;e)>0
\notRightarrow
I(Z;e\mid h)>0
}
\]

> *The physics can transfer while the epistemics are context-dependent.*  
> **A modality being informative in isolation does not imply that it is
> epistemically useful given the agent's existing belief state.**

中文：**世界模型不是把所有传感器都融合，而是根据当前 belief 中已经存在的信息，只吸收具有条件增量认识价值的模态。**

Door saturation (C.5 + V7D):

\[
\boxed{
h^{S}\ \text{already contains the information needed for current Door
structural-error judgment}
}
\]

\[
L(S)=0.441,\quad L(S{+}Z_{\mathrm{RGB}})=0.488,\quad L(S{+}Z_{\mathrm{shuf}})=0.470.
\]

Aligned RGB is worse than shuffled RGB. Extra modalities can be
**actively harmful** when redundant with \(h^{S}\).

The limit moved from representation to

\[
\boxed{\text{task/distribution does not require the additional modality}}
\]

Do **not** reopen GRU gates, occupancy penalties, \(\tau\) refits, the
V7C contact-proxy definition, Door CNN/RGB search, or **V7E**.

## H4 reading (frozen)

Contact sensing is **not** claimed to monotonically improve pooled
Brier (\(N\) was slightly better than \(S\) on mid/final). It **is**
claimed to carry task-relevant epistemic information:
\(B_{\mathrm{C0}}^{S}<B_{\mathrm{C0}}^{N}\) and C1-final Brier
\(^{S}<^{N}\), with C0 step-FPR \(0.013\).

\[
\boxed{
\text{contact sensing provides task-relevant epistemic information,
but its value is metric- and regime-dependent}
}
\]

## Sensor ladder (stopped on Door)

\[
\text{low-dimensional proprioceptive contact}
\;\not\longrightarrow\;
\text{Door tactile }Z
\;\not\longrightarrow\;
\text{Door RGB }Z
\]

Ledger: `REPORT/REP/R3_V7_EVIDENCE_CHAIN.md`.

## Next (not V7E)

**R4 — Contact-Rich Epistemic Domain**, first stage **R4-C0**:
`REPORT/REG/R4_C0_PREREG.md`. Ask
\(I(X_{\mathrm{tactile}};e_t\mid h^{S})>0\) on slip / friction /
distributed compliance, baseline \(h^{S}\) not \(h^{-c}\), **no encoder**
until observation necessity passes.
