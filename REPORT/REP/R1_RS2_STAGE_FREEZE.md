# R1-RS2 Stage Freeze — Physics Transfers, Epistemics Do Not

Date: 2026-08-16  
Status: **STAGE FROZEN**  
Does not change `RS2_GO=false`.  
Does not rewrite `RS1C_GO=true`.  
`RS2A_GO` and `RS2B_GO` remain diagnostic passes, not a rescue of Formal.

## One-line verdict

\[
\boxed{
\text{The physics can transfer while the epistemics are context-dependent.}
}
\]

中文：同一个世界的动力学表示可以跨交互机制迁移；认识量的含义取决于当前
具身历史，不会作为一把通用标尺自动迁移。后续精炼见
`REPORT/REP/R1_TRANSPORT_STAGE_FREEZE.md`。

## Frozen chain

\[
\boxed{
\begin{aligned}
RS2\text{-C0} &: \text{contact }J^\top f\text{ accounting closes}\\
RS2\text{-Formal} &: \text{frozen RS1C allocation does not transfer}\\
RS2A &: \text{detectability is not raw-exposure invariant}\\
RS2B &: \text{Mode-A }C\text{ is not intervention-invariant}
\end{aligned}
}
\]

Four non-contradictory facts:

\[
\boxed{
\begin{aligned}
\text{Physical transport} &: \textbf{yes}\\
\text{Structural detectability transport} &: \textbf{no, not automatically}\\
\text{Consequence transport} &: \textbf{no, not automatically}\\
\text{Frozen RS1C allocation transport} &: \textbf{no}
\end{aligned}
}
\]

## Two orthogonal Formal failures

\[
\boxed{
\begin{aligned}
C1 &: \text{evidence calibration failure (RS2A)}\\
C2 &: \text{consequence semantics failure (RS2B)}
\end{aligned}
}
\]

### RS2A — Can I see the mismatch?

\[
X_\phi^{contact}\approx X_\phi^{ModeA}
\not\Rightarrow
D_0^{contact}\approx D_0^{ModeA}.
\]

Interaction changes trajectory support and parameter-tangent geometry.
Detectability tracks tangent-visible exposure \(X_{\phi,\perp}\) more than
raw \(X_\phi\).

### RS2B — If I see it, do I still know how much it matters?

Not automatically. C2 is the clean case: 5/5 have
\(C_{\mathrm{ModeA}}<C_{\mathrm{tol}}\), while median contact incumbent loss
is \(0.141\) against C0 \(3.3\times10^{-5}\). The same \(\Delta C\)
functional stays \(\approx0\) on C2 contact, because oracle-drag and the
incumbent are equally wrong about latch.

C2 is therefore not “unseen mismatch.” It is:

\[
\boxed{
\text{saw the mismatch, but the old consequence functional does not know
why this mismatch matters}
}
\]

## \(D_0,C,V\) are not properties of the world

\[
D_0=D_0(M,\pi_{\mathrm{intervention}},D),\qquad
C=C(M,\pi_{\mathrm{intervention}},\mathcal L,\mathcal Q).
\]

They depend on the incumbent model, the intervention mechanism, the visited
state distribution, the evaluation queries, and the consequence functional.
They are not absolute labels of model mismatch. They are how mismatch
appears under a way of acting on the world and scoring forecasts.

## Threshold recalibration is closed

A mere scale drift would look like \(D_0^{contact}=k\,D_0^{ModeA}\) or a
monotone \(C_{\mathrm{contact}}=g(C_{\mathrm{ModeA}})\). Observed instead:

- the \(X_\phi\to D_0\) relation itself changes (RS2A);
- C1 Spearman \(\rho(C_{\mathrm{ModeA}},C_{\mathrm{contact}})=-0.115\)
  (ranking does not transport);
- C2 \(\Delta C\) is blind to latch harm.

Forbidden as a “fix”:

\[
\tau_{\mathrm{contact}}=0.3\,\tau_{\mathrm{ModeA}},
\qquad
C_{\mathrm{tol,contact}}=0.5\,C_{\mathrm{tol}}.
\]

Those are new domain-specific rulers, not transport.

## Relative vs absolute consequence

\[
\Delta C = L(\text{incumbent})-L(\text{candidate})
\]

asks how much a known candidate would help.

\[
L_{\mathrm{incumbent}}
\]

asks how wrong the incumbent already is versus the world.

C2:

\[
\Delta C\approx0\qquad\text{and}\qquad L_{\mathrm{incumbent}}\gg0,
\]

because both incumbent and oracle-drag are the wrong model class
(V1 pseudo-true parameter). Therefore:

\[
\boxed{
\text{candidate-relative utility}\not\Rightarrow\text{absolute adequacy}
}
\]

\[
\boxed{
\text{absence of a useful known revision}
\neq
\text{absence of consequential model error}
}
\]

Large typed inadequacy with no trusted in-library candidate must not be
sent to **tolerate** by \(\Delta C(\text{incumbent},\text{best known})\).

## Typed consequence (hypothesis only; not back-written into RS2B)

Analogous to \(E_{\mathrm{known}}\neq E_{\mathrm{unknown}}\):

\[
C_{\mathrm{known}}\neq C_{\mathrm{unknown}}
\]

is a **future** design hypothesis. Known-library operators may keep
\(\Delta C\). Outside-library evidence should use \(L_{\mathrm{incumbent}}\)
or another absolute/disagreement risk, not a fictional in-library oracle.
Do not treat this as an RS2B result.

## Strengthened principle

\[
\boxed{
\text{epistemic decision variables must be conditioned on the intervention regime}
}
\]

Policy is not \(\pi(D_0,C,V)\) in the abstract. More generally:

\[
\boxed{
\pi(D_0,C,V\mid\mathcal I)
}
\]

where \(\mathcal I\) is the intervention / interaction mechanism (direct
hinge torque, robot contact, grasp, push, collision, …). \(\mathcal I\)
changes the meaning of the statistics, not merely a domain ID.

## Permanently frozen (do not reopen as RS2 repair)

- retuning \(D_0\) cell thresholds or \(C_{\mathrm{tol}}\) on contact data;
- changing \(s^\star\), OSC scripts, or C0 geometry to make Formal pass;
- treating `RS2A_GO` / `RS2B_GO` as `RS2_GO=true`;
- rewriting `RS1C_GO`;
- installing support as a veto;
- opening RS1B.3.

## Status (strict)

\[
\boxed{
\begin{aligned}
RS1C\_GO &= true\\
RS2\text{-C0\_PASS} &= true\\
RS2\_GO &= false\\
RS2A\_GO &= true\\
RS2B\_GO &= true
\end{aligned}
}
\]

## Handoff

Do **not** rerun RS2. The next opened stage is **R1-RS3A only**
(`REPORT/REG/R1_RS3A_PREREG.md`): learner-visible transport-calibrated
structural evidence. RS3B/C remain locked. The question is

\[
\text{Can we construct a learner-visible evidence variable whose
semantics remain stable across intervention mechanisms?}
\]

not “fit contact-specific \(\tau\) and \(C_{\mathrm{tol}}\).”
